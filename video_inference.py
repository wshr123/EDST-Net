import time
import numpy as np
import torch
import tqdm
from core.utils import logging
from core.visualization.async_predictor import AsyncDemo, AsyncVis
from core.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from core.visualization.demo_loader import ThreadVideoManager, VideoManager
from core.visualization.predictor import ActionPredictor
from core.visualization.video_visualizer import VideoVisualizer
import timeit
logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES if len(cfg.DEMO.LABEL_FILE_PATH) != 0 else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    start_time = timeit.default_timer()
    num_task = 0
    # Start reading frames.
    total_time = 0.0
    total_frames = 0
    frame_provider.start()
    for able_to_read, task in frame_provider:
        total_frames += len(task.frames)
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1
        iter_time = model.put(task,total_time)
        if iter_time == None:
            continue
        total_time += iter_time
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue
    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue
    total_video_inference_time = timeit.default_timer() - start_time
    fps = (total_frames/total_video_inference_time)  #devide frame rate
    print("total frames",total_frames)
    print("total model process time",total_time)
    print("total video inference time", total_video_inference_time)
    print("fps",fps)

def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        cfg.DEMO.THREAD_ENABLE = False
        if cfg.DEMO.THREAD_ENABLE:
            frame_provider = ThreadVideoManager(cfg)
        else:
            frame_provider = VideoManager(cfg)
        for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
            frame_provider.display(task)
        frame_provider.join()
        frame_provider.clean()
