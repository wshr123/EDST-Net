
import core.utils.distribute as du
import core.utils.logging as logging
import core.utils.misc as misc
import core.dataset.loader as loader
from torch.utils.data import Dataset, DataLoader
from core.dataset import build_dataset, get_coco_api_from_dataset
import core.utils.checkpoint as cu
from core.dataset import loader
from core.utils.env import pathmgr
from core.utils.meters import AVAMeter, TestMeter
import core.visualization.tensorboard_vis as tb
import core.model.losses as losses
import numpy as np
import torch
from core.model.my_model import build
import os
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import pickle
import timeit
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
import os
import numpy as np
import torch.nn.functional as F
from train_net import CombinedDataset, combined_collate_fn
from core.utils.attn_visualization import *
logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, postprocess, writer=None):
    model.eval()
    test_meter.iter_tic()
    total_time = 0.0
    # for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(test_loader):
    for cur_iter, batch in enumerate(test_loader):
        ava_data = batch["ava"]
        coco_data = batch["coco"]
        inputs, labels, video_idx, time, meta, batch_length = ava_data
        detection_inputs, detection_labels = coco_data
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            image_ids = meta["image_id"]
            del meta["image_id"]
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                        # val[i] = val[i]
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            det = detection_inputs
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            det = det.to(device)
            # print(preds)
            ori_boxes = meta["ori_boxes"]
            # print(ori_boxes)
            metadata = meta["metadata"]
            # print(metadata)
            orig_target_sizes = torch.stack([t["orig_size"] for t in detection_labels], dim=0)
            start_time = timeit.default_timer()
            preds = model(inputs, det, meta["boxes"], orig_target_sizes, postprocess, detection_labels, batch_length, ori_boxes, mode="test")
            # preds = model(inputs, meta["boxes"])
            per_iter_time = timeit.default_timer() - start_time
            total_time += per_iter_time
            preds = preds['pred_actions']

            #==============================attention visualization================================
            #
            # visualize_feature_activation_simple(
            #     attn_dict,
            #     inputs,
            #     output_dir='attention_visualization',
            #     iter_num=cur_iter,
            #     colormap='magma',
            # )
            """
            'viridis' - matplotlib的默认颜色映射，从深蓝到黄色
            'plasma' - 从深蓝到红色，经过紫色
            'inferno' - 从黑到黄，经过红色
            'magma' - 从黑到白，经过紫色和粉红色
            'cividis' - 从蓝到黄，对色盲友好
            'hot' - 从黑到白，经过红色和黄色（热力图）
            'jet' - 经典彩虹色谱（从蓝到红）
            'coolwarm' - 从蓝到红（冷热对比）
            """
            # ==============================attention visualization===============================

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            metadata = metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
            # print(ori_boxes)
            test_meter.iter_toc()
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()
        try:
            test_meter.iter_toc()
        except:
            pass
        test_meter.iter_tic()
    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))
    logger.info("Finish test in: {}".format(total_time))
    test_meter.finalize_metrics()
    return test_meter,total_time


def test(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model, criterion, postprocessors = build(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        flops, params = 0.0, 0.0
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)
        use_detr_pretrian = True
        if use_detr_pretrian:

            checkpoint = torch.load('/media/zhong/1.0T/CVPR24/LW-DETR/output/$model_name/checkpoint_best_ema.pth',
                                    map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        # start_epoch = checkpoint_epoch + 1
        start_epoch = 0
        # Create video testing loaders.
        start_time = timeit.default_timer()
        test_ava_loader, test_ava_datsset = loader.construct_loader(cfg, "test")
        end_time = timeit.default_timer()
        dataset_time = end_time - start_time
        test_coco_dataset = build_dataset(image_set='val', args=cfg)
        combined_dataset_test = CombinedDataset(test_ava_datsset, test_coco_dataset)
        combined_loader_test = DataLoader(
            combined_dataset_test,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=combined_collate_fn,
            drop_last=False,
            pin_memory=True,
        )
        base_ds = get_coco_api_from_dataset(test_coco_dataset)

        start_time = timeit.default_timer()
        logger.info("Testing model for {} iterations".format(len(test_ava_loader)))
        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_ava_loader), cfg, mode="test")
        else:
            # print(test_loader.dataset.num_videos)
            assert (
                test_ava_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_ava_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                (
                    cfg.MODEL.NUM_CLASSES
                    if not cfg.TASK == "ssl"
                    else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM
                ),
                len(test_ava_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter, forward_time = perform_test(combined_loader_test, model, test_meter, cfg, postprocessors, writer)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()
    end_time = timeit.default_timer()
    total_time = end_time - start_time + dataset_time
    key_frame = 1332
    per_clip_latency = forward_time / key_frame
    # fps = total_frame / total_time
    print('forward time', forward_time)
    print('total time:', total_time)
    print("latency:",per_clip_latency)
    # print('fps:', fps)
    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".format(
                view, cfg.TEST.NUM_SPATIAL_CROPS
            )
        )
        result_string_views += "_{}a{}" "".format(view, test_meter.stats["top1_acc"])

        result_string = (
            "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
            "".format(
                params / 1e6,
                flops,
                view,
                test_meter.stats["top1_acc"],
                test_meter.stats["top5_acc"],
                misc.gpu_mem_usage(),
                flops,
            )
        )

        logger.info("{}".format(result_string))
    logger.info("{}".format(result_string_views))
    return result_string + " \n " + result_string_views

