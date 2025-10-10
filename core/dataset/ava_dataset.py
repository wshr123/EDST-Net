#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import cv2
import numpy as np
import torch

from . import (
    ava_helper as ava_helper,
    cv2_transform as cv2_transform,
    transform as transform,
    utils as utils,
)
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP
        self._load_data(cfg)




    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))
        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(cfg, mode=self._split)
        # print(len(self._image_paths))
        # print(len(boxes_and_labels))
        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        import numpy as np

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )

        elif self._split == "test":

            # ====== different ligth sim======
            # import os, cv2, numpy as np
            #
            # def _thirds_mode(idx_in_video: int, total: int) -> str:
            #     t1 = total // 3
            #     t2 = (2 * total) // 3
            #     if idx_in_video < t1:
            #         return "dark"
            #     elif idx_in_video < t2:
            #         return "none"
            #     else:
            #         return "bright"
            #
            # def _apply_exposure_L_add_uint8(bgr_u8: np.ndarray, mode: str) -> np.ndarray:
            #     lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
            #     L = lab[:, :, 0].astype(np.int16)
            #     if mode == "dark":
            #         delta = -40  
            #     elif mode == "bright":
            #         delta = +40  
            #     else:
            #         delta = 0
            #     L = np.clip(L + delta, 0, 255).astype(np.uint8)
            #     lab[:, :, 0] = L
            #     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            #
            # if not hasattr(self, "_save_root_hr"):
            #     self._save_root_hr = "./debug_exposure/by_path_hr"
            #     os.makedirs(self._save_root_hr, exist_ok=True)
            # if not hasattr(self, "_by_path_idx"):
            #     self._by_path_idx = {}  # {video_idx: {abs_path: index_in_video}}
            #
            # vid = getattr(self, "_current_video_idx", None)
            # if vid is not None:
            #     all_paths = self._image_paths[vid]  
            #     total_frames_video = len(all_paths)
            #     if vid not in self._by_path_idx:  
            #         self._by_path_idx[vid] = {p: i for i, p in enumerate(all_paths)}
            #     seq = utils.get_sequence(
            #         self._current_center_idx,
            #         self._seq_len // 2,
            #         self._sample_rate,
            #         num_frames=total_frames_video,
            #     )
            #     image_paths_clip = [all_paths[f - 1] for f in seq]
            #
            #     if len(image_paths_clip) > 0:
            #         video_folder_name = os.path.basename(os.path.dirname(image_paths_clip[0]))
            #         out_dir = os.path.join(self._save_root_hr, video_folder_name)
            #         os.makedirs(out_dir, exist_ok=True)
            #
            #     processed = []
            #     for im_u8, abs_path in zip(imgs, image_paths_clip):
            #         idx_in_video = self._by_path_idx[vid].get(abs_path, None)
            #         if im_u8.dtype != np.uint8:
            #             im_u8 = np.clip(im_u8, 0, 255).astype(np.uint8)
            #         if im_u8.ndim == 3 and im_u8.shape[0] in (1, 3) and im_u8.shape[2] not in (1, 3):
            #             im_u8 = np.transpose(im_u8, (1, 2, 0))
            #
            #         if idx_in_video is None:
            #             im_proc = im_u8  
            #         else:
            #             mode = _thirds_mode(idx_in_video, total_frames_video)
            #             im_proc = _apply_exposure_L_add_uint8(im_u8, mode)
            #
            #         if len(image_paths_clip) > 0:
            #             save_path = os.path.join(out_dir, os.path.basename(abs_path))
            #             try:
            #                 cv2.imwrite(save_path, im_proc)
            #             except Exception as e:
            #                 print(f"[exposure-save warn] fail to write {save_path}: {e}")
            #
            #         processed.append(im_proc)
            #
            #     imgs = processed  


            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )



        else:
            raise NotImplementedError("Unsupported split mode {}".format(self._split))

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(imgs, self._crop_size, boxes=boxes)

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )
            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError("{} split not supported yet!".format(self._split))

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(boxes, self._crop_size, self._crop_size)

        return imgs, boxes

    def _to_uint8_hwc(self, img: np.ndarray) -> np.ndarray:
        """确保输入是连续的 uint8 HWC（BGR）。支持 float[0,1] 或 CHW 转换。"""
        arr = np.asarray(img)
        # 如果是 CHW（C,H,W）且 C 在前，把它转成 HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

        # 若是 float 且在 [0,1]，转成 uint8
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                # 先clip再量化，避免溢出
                arr = np.clip(arr, 0.0, 1.0) * 255.0
                arr = np.round(arr).astype(np.uint8)
            else:
                # 其它整型/类型，直接clip到[0,255]再转
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        # 保证连续
        return np.ascontiguousarray(arr)

    def _apply_gamma_uint8(self, img, gamma: float) -> np.ndarray:
        img_u8 = self._to_uint8_hwc(img)
        if abs(gamma - 1.0) < 1e-6:
            return img_u8
        inv = 1.0 / max(gamma, 1e-6)
        # 单通道 LUT，长度 256，uint8，连续
        lut = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(img_u8, lut)  # (H,W,C) 每个通道都会应用 LUT
        return out

    def adjust_exposure_uint8(self,img_bgr: np.ndarray, mode: str,
                              dark_cfg=None, bright_cfg=None) -> np.ndarray:
        """
        img_bgr: HWC, BGR（容错：若不是，会自动转成 HWC uint8）
        mode: "dark" | "none" | "bright"
        """
        img_bgr = self._to_uint8_hwc(img_bgr)
        if mode == "none":
            return img_bgr

        if mode == "dark":
            gamma = (dark_cfg or {}).get("gamma", 1.05)
            alpha = (dark_cfg or {}).get("alpha", 0.85)
        else:  # "bright"
            gamma = (bright_cfg or {}).get("gamma", 0.95)
            alpha = (bright_cfg or {}).get("alpha", 1.15)

        out = self._apply_gamma_uint8(img_bgr, gamma)
        # 线性增益（alpha>1 亮，<1 暗）；保持 uint8
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=0)
        return out

    def thirds_mode(self, center_idx: int, total_frames: int) -> str:
        t1 = total_frames // 3
        t2 = (2 * total_frames) // 3
        if center_idx < t1:
            return "dark"
        elif center_idx < t2:
            return "none"
        else:
            return "bright"

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            time index (zero): The time index is currently not supported for AVA.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(idx, tuple):
            idx, self._num_yielded = idx
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                idx, short_cycle_idx = idx

        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        #todo  save time information
        self._current_video_idx = video_idx
        self._current_center_idx = center_idx
        self._current_total_frames = len(self._image_paths[video_idx])

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame-1] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        )
        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(imgs, boxes=boxes)
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(imgs, boxes=boxes)

        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1

        imgs = utils.pack_pathway_output(self.cfg, imgs)    #这里分开采样了slow,fast
        metadata = [[video_idx, sec]] * len(boxes)
        img_id = image_paths
        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
            "image_id": image_paths
        }

        return imgs, label_arrs, idx, torch.zeros(1), extra_data
