# 🐮 EDST-Net

Source code of  
**_“EDST-Net: An Efficient Dual-Stream Parallel Framework for Spatio-Temporal Action Detection of Multi-Cattle Behaviors”_**

<p align="center">
  <img src="all_model.png" alt="EDST-Net Architecture" width="800">
</p>

---

Notice:

we have update Preparation in Oct.3

## 📋 Table of Contents
1. [Installation](#1-installation)
2. [Preparation](#2-preparation)
   - [Data Preparation](#data-preparation)
   - [Model Preparation](#model-preparation)
3. [Inference Demo with Pretrained Model](#3-inference-demo-with-pretrained-model)
4. [Training](#4-train)
5. [Testing](#5-test)
6. [References](#6-references)
7. [Citation](#7-citation)

---

## 1️⃣ Installation

Please find installation instructions for PyTorch and EDST-Net in  
👉 [INSTALL.md](INSTALL.md)

---

## 2️⃣ Preparation

### 🧩 Data Preparation

Follow the instructions in  
👉 [DATASET.md](DATASET.md)  
to prepare the **AVA-format datasets** used for training and evaluation.

---

#### 📦 Step-by-Step Dataset Annotation Pipeline

👉 [step_by_step_dataset.md](step_by_step_dataset.md)  

> 🛠️ This section provides a detailed guide for building annotations matching the EDST-Net input format.

As an alternative, you can directly download our prepared annotations (links will be available soon).

```text
===========  Under Construction  ===========
✅ The full step-by-step introduction will be available before **Oct. 7**
✅ CVB and CVB-I annotation files will be uploaded to this repository
============================================

### 🧠 Model Preparation

We pretrain the **LW-DETR** on the CVB dataset.  
Please download the pretrained weights and place them in the `weights/` folder.

🔗 [LW-DETR Pretrained Model](https://drive.google.com/file/d/1VAyJ9jrJex7s_cmNKvrtINqMznVG9Xit/view?usp=sharing)

Then modify the configuration file:

```yaml
CHECKPOINT_LWDETR: "path/to/your/lw_detr_checkpoint.pth"
```

If you prefer to train LW-DETR yourself, refer to the official repo:  
👉 [LW-DETR GitHub Repository](https://github.com/Atten4Vis/LW-DETR)

---

We use **Kinetics-400 pretrained weights** for our temporal backbone.  
Please download and place them in `weights/`:

🔗 [Temporal Backbone (X3D-L)](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_l.pyth)

Then update in your config file:

```yaml
CHECKPOINT_FILE_PATH: "path/to/your/x3d_l.pyth"
```

---

## 3️⃣ Inference Demo with Pretrained Model

We provide an **inference demo** for visualizing custom input videos using pretrained weights.

### ▶️ Steps:

1. **Download pretrained weights:**  
   [EDST-Net Pretrained Models](https://drive.google.com/drive/folders/1EYcWb0f4WfnMKLIAGcNzNOmwZjogwjNb?usp=drive_link)

2. **Set the config flags:**
   ```yaml
   DEMO.ENABLE: True
   TRAIN.ENABLE: False
   TEST.ENABLE: False
   ```

3. **Modify paths in config:**
   ```yaml
   LABEL_FILE_PATH: "path/to/label.json"
   INPUT_VIDEO: "path/to/your_video.mp4"
   OUTPUT_FILE: "path/to/output_demo.mp4"
   ```

4. **Run the demo:**
   ```bash
   python run_net.py --cfg your/file_path/edst.yaml
   ```

The model will output a visualization video with bounding boxes and action labels for each detected cattle behavior.

---

## 4️⃣ Train

To start training from scratch, enable training in your config file:

```yaml
TRAIN.ENABLE: True
```

Then run:

```bash
python run_net.py --cfg your/file_path/edst.yaml
```

> 💡 You can modify batch size, learning rate, and training epochs in the YAML config file as needed.

---

## 5️⃣ Test

To evaluate or test the model:

```yaml
TEST.ENABLE: True
```

Then run:

```bash
python run_net.py --cfg your/file_path/edst.yaml
```

For pretrained checkpoints, please refer to  
👉 [Inference Demo with Pretrained Model](#3-inference-demo-with-pretrained-model)

---

## 6️⃣ References

Our work builds upon the following open-source projects:

- [SLOWFAST](https://github.com/facebookresearch/SlowFast)
- [LW-DETR](https://github.com/Atten4Vis/LW-DETR)

---

## 7️⃣ Citation

If you find this work useful in your research, please kindly consider citing our paper:

```bibtex
@article{zhong2025edstnet,
  title={EDST-Net: An Efficient Dual-Stream Parallel Framework for Spatio-Temporal Action Detection of Multi-Cattle Behaviors},
  author={Zhong, Huiyu and Su, Daobilige and Qiao, Yongliang and Yang, Zhe and Wang, Xuechang and Wang, Qingjie and Zhang, Xinyue},
  journal={Under Review},
  year={2025}
}
```

---

<p align="center">
  ⭐ Star this repo if you find it helpful!  
  🐄 Contributions and issues are always welcome.
</p>
