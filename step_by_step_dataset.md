CVB Dataset for EDST-Net Processing and Model Training Guide

This repository provides a step-by-step workflow for processing the CVB dataset and training models, with special handling for Category 12 (rare class) to ensure that both validation and test sets contain sufficient samples.
The optimized pipeline improves dataset balance and model generalization.

📁 Repository Structure

All data processing scripts are located in:

utils/dataset_transform_tools/


🚀 Overview of the Optimized Pipeline

Processing order optimized: Fill frame gaps → Handle Category 12 distribution → Split train/val sets → Generate annotations → Train models.

Step	Task	Description
0	Fill missing frames	Ensure each video folder has 450 frames
1	Handle Category 12 imbalance	Guarantee samples appear in validation/test sets
2	Split val set from train set.	Create balanced training and validation splits
3	Generate prediction-box list	For evaluation and testing
4	Generate frame-level lists	Build frame index files for training
5	Generate video-name lists	For COCO conversion scripts
6	Convert to COCO format ompatible with model input format
7	Update model config	Link dataset paths and modes

🧩 Detailed Workflow
0️⃣ Fill Missing Frames Using fill_imgs.py

Script: utils/dataset_transform_tools/fill_imgs.py
Purpose:
Some CVB dataset video folders contain fewer than 450 frames.
This script duplicates the last frame until each folder reaches the same frame count (default: 450).

python utils/dataset_transform_tools/fill_imgs.py \
  --root_dir path/to/frames_directory \
  --target_frames 450


Arguments

Argument	Description	Default
--root_dir	Root folder where each sub-directory corresponds to a video	required
--target_frames	Target frame count; folders with fewer frames will be padded	450

⚠️ Check the console log after execution to verify the number of padded frames.
Run once only, before any dataset split.

1️⃣ Handle Category 12 Distribution

Script: utils/dataset_transform_tools/share_class12_to_val.py
Input: ava_train_set.csv
Output: videos_with_action_12.txt

python utils/dataset_transform_tools/share_class12_to_val.py \
  --csv_file path/to/ava_train_set.csv \
  --output_file path/to/videos_with_action_12.txt


Manual step:

Select several videos from videos_with_action_12.txt to move into the test set.

Remove those videos from the original training file.

Save the result as ava_train_set_modified.csv.

⚠️ This step must be done before any dataset split.

2️⃣ Split Train and Validation Sets

Script: utils/dataset_transform_tools/split_train_val.py
Input: ava_train_set_modified.csv
Output:

ava_val_set.csv (validation/test)

new_train_set.csv (training)

python utils/dataset_transform_tools/split_train_val.py \
  --input_file path/to/ava_train_set_modified.csv \
  --output_file_10_percent path/to/ava_val_set.csv \
  --output_file_remaining path/to/new_train_set.csv

3️⃣ Generate Prediction-Box List (for Test Set)

Script: utils/dataset_transform_tools/create_predict_boxes_list.py
Input: ava_val_set.csv
Output: ava_test_predit_boxes.csv

python utils/dataset_transform_tools/create_predict_boxes_list.py \
  --input_file path/to/ava_val_set.csv \
  --output_file path/to/ava_test_predit_boxes.csv

4️⃣ Generate Frame-Level Lists

Script: utils/dataset_transform_tools/create_frame_list.py
Inputs:

new_train_set.csv

ava_val_set.csv

Outputs:

train_frame.csv

test_frame.csv

# Training frames
python utils/dataset_transform_tools/create_frame_list.py \
  --input_file path/to/new_train_set.csv \
  --output_file path/to/train_frame.csv

# Testing frames
python utils/dataset_transform_tools/create_frame_list.py \
  --input_file path/to/ava_val_set.csv \
  --output_file path/to/test_frame.csv

5️⃣ Generate Video-Name Lists

Script: utils/dataset_transform_tools/get_video_names.py
Inputs:

new_train_set.csv

ava_val_set.csv

Outputs:

ava_file_names_train.txt

ava_file_names_test.txt

# Training video list
python utils/dataset_transform_tools/get_video_names.py \
  --input_csv_path path/to/new_train_set.csv \
  --output_txt_path path/to/ava_file_names_train.txt

# Testing video list
python utils/dataset_transform_tools/get_video_names.py \
  --input_csv_path path/to/ava_val_set.csv \
  --output_txt_path path/to/ava_file_names_test.txt

6️⃣ Convert CSVs to COCO Format

Script: utils/dataset_transform_tools/csv2coco.py
Inputs:

new_train_set.csv

ava_val_set.csv

ava_file_names_train.txt

ava_file_names_test.txt

Outputs:

train_min.json

test_min.json

# Training annotations
python utils/dataset_transform_tools/csv2coco.py \
  --csv_path path/to/new_train_set.csv \
  --movie_list path/to/ava_file_names_train.txt \
  --img_root path/to/frames_directory \
  --json_path path/to/train.json \
  --min_json_path path/to/train_min.json

# Testing annotations
python utils/dataset_transform_tools/csv2coco.py \
  --csv_path path/to/ava_val_set.csv \
  --movie_list path/to/ava_file_names_test.txt \
  --img_root path/to/frames_directory \
  --json_path path/to/test.json \
  --min_json_path path/to/test_min.json

7️⃣ Update Model Configuration

File: core/dataset/coco.py

# Around line 175
PATHS = {
    "train": ("/absolute/path/to/frames_directory", 
              "/absolute/path/to/train_min.json"),
    "val": ("/absolute/path/to/frames_directory", 
            "/absolute/path/to/test_min.json"),
    "test": ("/absolute/path/to/frames_directory", 
             "/absolute/path/to/test_min.json")
}

# Around line 66
root = Path("/absolute/path/to/your/dataset_root")

⚙️ Model Training and Evaluation Configuration
if training_from_scratch:  # Use X3D-L pretrained weights
    checkpoint["model_state"] = {
        "temporal_backbone." + k: v 
        for k, v in checkpoint["model_state"].items()
    }
else:  # For testing or inference
    # Comment out the code above

Scenario	Action
✅ Training from scratch	Enable prefix addition (X3D-L initialization)
❌ Testing or inference	Disable prefix addition (load complete EDST-Net directly)
🩹 Troubleshooting
1️⃣ Category 12 Handling Not Working

Ensure the Category 12 processing step is completed before dataset splitting.

Verify videos_with_action_12.txt videos were correctly moved to the test set.

Re-generate the test JSON file.

2️⃣ Path Configuration Errors

Error: FileNotFoundError or empty dataset.

PATHS = {
    "train": ("/actual/path/to/frames", "/actual/path/to/train_min.json"),
}
root = Path("/actual/dataset/root")

3️⃣ Import Errors (Missing PyTorchVideo)
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .

📂 File Structure Overview
File / Folder	Purpose
frames_directory/	Root folder containing all video frames
*_min.json	Minimal COCO annotation files
ava_file_names_*.txt	Video-name index lists
*_frame.csv	Frame-path lists for training
ava_test_predit_boxes.csv	Pre-computed detection boxes
utils/dataset_transform_tools/	Contains all preprocessing scripts
💡 Best Practices

✅ Always use absolute paths

✅ Keep directory structures consistent

✅ Follow the correct order:
fill frames → handle Category 12 → split → generate files → train


