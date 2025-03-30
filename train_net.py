from core.model.my_model import build
from torch.utils.data import Dataset, DataLoader
import math
import pprint
from thop import profile
import numpy as np
from core.dataset import build_dataset, get_coco_api_from_dataset
import core.model.losses as losses
import core.model.optimizer as optim
import core.utils.checkpoint as cu
import core.utils.distributed as du
import core.utils.logging as logging
import core.utils.metrics as metrics
import core.utils.misc as misc
import core.visualization.tensorboard_vis as tb
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from core.dataset import loader
from core.dataset.mixup import MixUp
# from core.model import build_model
from core.model.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from core.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from core.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    postprocess,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    # for cur_iter, (inputs, labels, index, time, meta) in enumerate(train_loader):
        # print(meta)
        # Transfer the data to the current GPU device.
    for cur_iter, batch in enumerate(train_loader):
        ava_data = batch["ava"]
        coco_data = batch["coco"]
        inputs, labels, index, time, meta, batch_length = ava_data
        detection_inputs, detection_labels = coco_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detection_inputs = detection_inputs.to(device)
        for d in detection_labels:
            d['image_path'] = d['image_path'][0]
        detection_labels =[{k: (v.to(device) if k != 'image_path' else v) for k, v in t.items()} for t in detection_labels]
        orig_target_sizes = torch.stack([t["orig_size"] for t in detection_labels], dim=0)
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            del meta["image_id"]
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        ori_boxes = meta["ori_boxes"]
        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()

            if cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                (
                    model,
                    preds,
                    partial_loss,
                    perform_backward,
                ) = contrastive_forward(
                    model, cfg, inputs, index, time, epoch_exact, scaler
                )
            elif cfg.DETECTION.ENABLE:
                preds = model(inputs, detection_inputs, meta, orig_target_sizes, postprocess,
                              detection_labels, batch_length, ori_boxes)
            elif cfg.MASK.ENABLE:
                preds, labels = model(inputs)
            else:
                preds = model(inputs)
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                labels = torch.zeros(
                    preds.size(0), dtype=labels.dtype, device=labels.device猴

                )

            if cfg.MODEL.MODEL_NAME == "ContrastiveModel" and partial_loss:
                loss = partial_loss
            else:
                # Compute the loss.
                loss = loss_fun(preds['pred_actions'], labels)

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())
        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        if update_param:
            scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                loss, grad_norm = (
                    loss.item(),
                    grad_norm.item(),
                )
            elif cfg.MASK.ENABLE:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                    if loss_extra:
                        loss_extra = du.all_reduce(loss_extra)
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    0.0,
                    0.0,
                )
                if loss_extra:
                    loss_extra = [one_loss.item() for one_loss in loss_extra]
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm, top1_err, top5_err = du.all_reduce(
                        [loss.detach(), grad_norm, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                grad_norm,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                loss_extra,
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
    del inputs

    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, train_loader, postprocess, writer):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    # for batch in val_loader:
    #     print(batch)
    #     break
    # for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(val_loader):
    for cur_iter, batch in enumerate(val_loader):
        ava_data = batch["ava"]
        coco_data = batch["coco"]
        inputs, labels, video_idx, time, meta, batch_length = ava_data
        detection_inputs, detection_labels = coco_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detection_inputs = detection_inputs.to(device)
        for d in detection_labels:
            d['image_path'] = d['image_path'][0]
        detection_labels = [{k: (v.to(device) if k != 'image_path' else v) for k, v in t.items()} for t in
                            detection_labels]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detection_labels = [{k: (v.to(device) if k != 'image_path' else v) for k, v in t.items()} for t in
                                detection_labels]
        orig_target_sizes = torch.stack([t["orig_size"] for t in detection_labels], dim=0)

        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            image_ids = meta["image_id"]
            del meta["image_id"]
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = video_idx
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]
            # Compute the predictions.
            # preds = model(inputs, meta["boxes"])

            preds = model(inputs, detection_inputs, meta, orig_target_sizes, postprocess,
                          detection_labels, batch_length, ori_boxes)
            if cfg.NUM_GPUS:
                preds = preds['pred_actions']
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if not cfg.CONTRASTIVE.KNN_ON:
                    return
                train_labels = (
                    model.module.train_labels
                    if hasattr(model, "module")
                    else model.train_labels
                )
                yd, yi = model(inputs, index, time)
                K = yi.shape[1]
                C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                probs = torch.mul(
                    retrieval_one_hot.view(batch_size, -1, C),
                    yd_transform.view(batch_size, -1, 1),
                )
                preds = torch.sum(probs, 1)
            else:
                preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                if cfg.DATA.IN22k_VAL_IN1K != "":
                    preds = preds[:, :1000]
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars({"Val/mAP": val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [label.clone().detach() for label in val_meter.all_labels]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, target in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters, )


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model, criterion, postprocesser  = build(cfg)
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )

class CombinedDataset(Dataset):
    def __init__(self, ava_dataset, coco_dataset):
        super().__init__()
        self.ava_dataset = ava_dataset
        self.coco_dataset = coco_dataset
        self.ava_imageid_to_idx = {}
        load_method = "acc"
        if load_method =="normal":
            for idx in range(len(self.ava_dataset)):
                data_ava = self.ava_dataset[idx]
                imgs, label_arrs, id, det, extra_data = data_ava
                ava_key = extra_data['image_id'][len(extra_data['image_id'])//2]
                self.ava_imageid_to_idx[ava_key] = idx
            self.coco_imageid_to_idx = {}
            for idx in range(len(self.coco_dataset)):
                data_coco = self.coco_dataset[idx]
                imgs, targets = data_coco
                coco_key = targets['image_path'][0]
                self.coco_imageid_to_idx[coco_key] = idx
            self.shared_keys = sorted(
                list(set(self.ava_imageid_to_idx.keys()) & set(self.coco_imageid_to_idx.keys()))
            )
        elif load_method == "acc": # accelerate dateaset prepare process
            from concurrent.futures import ThreadPoolExecutor
            def process_ava_data(idx):
                data_ava = self.ava_dataset[idx]
                imgs, label_arrs, id, det, extra_data = data_ava
                ava_key = extra_data['image_id'][len(extra_data['image_id'])//2]
                return ava_key, idx
            def process_coco_data(idx):
                data_coco = self.coco_dataset[idx]
                imgs, targets = data_coco
                coco_key = targets['image_path'][0]
                return coco_key, idx
            with ThreadPoolExecutor() as executor:
                ava_results = list(executor.map(process_ava_data, range(len(self.ava_dataset))))
            self.ava_imageid_to_idx = {key: idx for key, idx in ava_results}
            with ThreadPoolExecutor() as executor:
                coco_results = list(executor.map(process_coco_data, range(len(self.coco_dataset))))
            self.coco_imageid_to_idx = {key: idx for key, idx in coco_results}
            self.shared_keys = sorted(set(self.ava_imageid_to_idx.keys()) & set(self.coco_imageid_to_idx.keys()))
        check_unexpected_keys = False
        if check_unexpected_keys:
            ava_unmatched_keys = set(self.ava_imageid_to_idx.keys()) - set(self.coco_imageid_to_idx.keys())
            coco_unmatched_keys = set(self.coco_imageid_to_idx.keys()) - set(self.ava_imageid_to_idx.keys())
            print(f"Keys in AVA but not in COCO: {ava_unmatched_keys}")
            print(f"Keys in COCO but not in AVA: {coco_unmatched_keys}")

    def __len__(self):
        return len(self.shared_keys)

    def __getitem__(self, idx):
        image_path_key = self.shared_keys[idx]
        ava_idx = self.ava_imageid_to_idx[image_path_key]
        coco_idx = self.coco_imageid_to_idx[image_path_key]
        data_ava = self.ava_dataset[ava_idx]
        data_coco = self.coco_dataset[coco_idx]

        return {
            "ava": data_ava,
            "coco": data_coco,
        }

def combined_collate_fn(batch):
    from core.dataset.loader import detection_collate
    from core.utils.misc import collate_fn

    ava_part = [b["ava"] for b in batch]
    coco_part = [b["coco"] for b in batch]
    ava_collated = detection_collate(ava_part)
    coco_collated = collate_fn(coco_part)

    return {
        "ava": ava_collated,
        "coco": coco_collated,
    }


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, criterion, postprocesser = build(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            cfg,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0
    # Create the video train and val loaders.
    ava_train_loader, ava_train_dataset = loader.construct_loader(cfg, "train")
    ava_val_loader, ava_val_dataset = loader.construct_loader(cfg, "val")
    coco_train_dataset = build_dataset(image_set='train', args=cfg)
    coco_val_dataset = build_dataset(image_set='val', args=cfg)
    sampler_train = torch.utils.data.RandomSampler(coco_train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(coco_val_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.TRAIN.BATCH_SIZE, drop_last=True)
    # cocodataloader_train = DataLoader(cocotrain_dataset, batch_sampler=batch_sampler_train,
    #                                collate_fn=misc.collate_fn, num_workers=cfg.DATA_LOADER.NUM_WORKERS) #return img_path:absolute path/219/img_00150.jpg
    # cocodataloader_val = DataLoader(cocoval_dataset, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val,
    #                              drop_last=False, collate_fn=misc.collate_fn,
    #                              num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    dataset_train = CombinedDataset(ava_train_dataset, coco_train_dataset)
    dataset_val = CombinedDataset(ava_val_dataset, coco_val_dataset)
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=combined_collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=combined_collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    base_ds = get_coco_api_from_dataset(coco_val_dataset)
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(ava_train_loader)
        else:
            model.init_knn_labels(ava_train_loader)

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(ava_train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(ava_val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(ava_train_loader), cfg)
        val_meter = ValMeter(len(ava_val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            loader_train = loader.construct_loader(cfg, "train")
            loader.shuffle_dataset(loader_train, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    loader_train,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)
        calculate_gflops = False
        if calculate_gflops:
            #to test params and gflops, please set batch size to 1
            # clac the model gflops and params
            for cur_iter, batch in enumerate(loader_train):
                ava_data = batch["ava"]
                coco_data = batch["coco"]
                inputs, labels, index, time, meta, batch_length = ava_data
                detection_inputs, detection_labels = coco_data
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                detection_inputs = detection_inputs.to(device)
                for d in detection_labels:
                    d['image_path'] = d['image_path'][0]
                detection_labels = [{k: (v.to(device) if k != 'image_path' else v) for k, v in t.items()} for t in
                                    detection_labels]
                orig_target_sizes = torch.stack([t["orig_size"] for t in detection_labels], dim=0)
                if cfg.NUM_GPUS:
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            if isinstance(inputs[i], (list,)):
                                for j in range(len(inputs[i])):
                                    inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                            else:
                                inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    if not isinstance(labels, list):
                        labels = labels.cuda(non_blocking=True)
                        index = index.cuda(non_blocking=True)
                        time = time.cuda(non_blocking=True)
                    del meta["image_id"]
                    for key, val in meta.items():
                        if isinstance(val, (list,)):
                            for i in range(len(val)):
                                val[i] = val[i].cuda(non_blocking=True)
                        else:
                            meta[key] = val.cuda(non_blocking=True)

                meta = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in meta.items()}
                # for det in detection_labels:

                ori_boxes = meta["ori_boxes"]
                labels = labels[:1]
                flops, params = 0.0, 0.0
                batch_length = [1]
                flops, params = profile(model, inputs=(
                    inputs, detection_inputs, meta, orig_target_sizes, postprocesser,
                    detection_labels, batch_length, ori_boxes))
                gflops = flops / 1e9
                params = misc.log_model_info(model, cfg, use_train_input=True)
                print(f"Params: {params:,}, GFLOPs: {gflops:.4f}")
                break
        else:
                flops, params = 0.0, 0.0
        # Shuffle the dataset.
        loader.shuffle_dataset(loader_train, cur_epoch)
        if hasattr(loader_train.dataset, "_set_epoch_num"):
            loader_train.dataset._set_epoch_num(cur_epoch)
        # eval_epoch(loader_val, model, val_meter, start_epoch, cfg, loader_train, postprocesser, writer)#todo do eval test
        # Train for one epoch.
        epoch_timer.epoch_tic()

        cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
        train_epoch(
            loader_train,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            postprocesser,
            writer,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(loader_train):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(loader_train):.2f}s in average."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
        )

        # Save a checkpoint.
        is_checkp_epoch = True
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                loader_val,
                model,
                val_meter,
                cur_epoch,
                cfg,
                loader_train,
                postprocesser,
                writer,
            )
    if (
        start_epoch == cfg.SOLVER.MAX_EPOCH and not cfg.MASK.ENABLE
    ):  # final checkpoint load
        eval_epoch(loader_val, model, val_meter, start_epoch, cfg, loader_train, postprocesser, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            (
                epoch_timer.median_epoch_time() / 60.0
                if len(epoch_timer.epoch_times)
                else 0.0
            ),
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))

    return result_string
