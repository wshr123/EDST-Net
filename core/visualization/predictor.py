#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
from PIL import Image
import cv2
import numpy as np
import core.utils.checkpoint as cu
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from core.dataset import cv2_transform
from core.model.my_model import build
from core.utils import logging
from core.visualization.utils import process_cv2_inputs
import core.dataset.transforms as T
import timeit
from core.dataset.utils import tensor_normalize
logger = logging.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = torch.cuda.current_device() if gpu_id is None else gpu_id

        # Build the video model and print model statistics.
        self.model, criterion, self.postprocess = build(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        self.cfg = cfg
        # video_input = torch.randn(1, 3, 16, 356, 356).to("cuda")  # (batch, channels, frames, height, width)
        # detection_input = torch.randn(1, 3, 640, 640).to("cuda")
        # #warm up
        # with torch.no_grad():
        #     for _ in range(10):
        #         _ = self.model(video_input, detection_input)
        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        # if self.cfg.DETECTION.ENABLE:
        #     task = self.object_detector(task)
        frames = task.frames
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        #==================process detection inputs======================
        t = self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        t_mid = t // 2
        mid_frame = frames[t_mid]
        mid_frame = np.array(mid_frame, dtype=np.uint8)
        mid_frame = np.transpose(mid_frame, (0, 1, 2))
        mid_frame = Image.fromarray(mid_frame)
        # mid_frame.show()
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transforms = T.Compose([
            T.SquareResize([640]),
            normalize,
        ])
        mid_frame = transforms(mid_frame)[0]
        mid_frame = mid_frame.unsqueeze(0)
        #======================process action inputs======================
        frames = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame) for frame in frames
        ]
        inputs = process_cv2_inputs(frames, self.cfg)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )
            if isinstance(mid_frame, (list,)):
                for i in range(len(mid_frame)):
                    mid_frame[i] = mid_frame[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                mid_frame = mid_frame.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )
        preds = self.model(inputs, mid_frame, postprocess=self.postprocess, mode='demo')
        if preds == None:
            return None
        #out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_actions': fuse_feat}
        if self.cfg.NUM_GPUS:
            bboxes = preds['pred_boxes']
            preds = preds['pred_actions']
            bboxes = bboxes.cpu()
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.detach().cpu()

        preds = preds.detach()
        task.add_action_preds(preds)
        if bboxes is not None:
            task.add_bboxes(bboxes[:, 1:])

        return task


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis

    def put(self, task, total_time=None):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        start_time = timeit.default_timer()
        task = self.predictor(task)
        per_iter_time = timeit.default_timer() - start_time
        if task == None:
            return None
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)
        return per_iter_time



    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")
        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        middle_frame = task.frames[len(task.frames) // 2]
        outputs = self.predictor(middle_frame)
        # Get only human instances
        mask = outputs["instances"].pred_classes == 19  #todo
        pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        task.add_bboxes(pred_boxes)

        return task
