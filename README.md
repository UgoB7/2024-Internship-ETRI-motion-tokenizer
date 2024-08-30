# Motion Token Learning

## Authors
- Youngwoo
- Ugo (intern)


## Overview
The goal of this project is to train a tokenizer for motion data by leveraging multiple motion datasets to create a more generalized tokenizer. This internship involves exploring existing datasets, retargeting joint structures from one to another, and training a new model for motion tokenization.

## Task/Status

### Done
- **Explore BEAT Dataset**  
  URL: [BEAT Dataset](https://pantomatrix.github.io/BEAT/)
  
- **Explore motion tokenizing codebases and choose one for future development**  
  Selected Codebase: [MotionGPT](https://github.com/OpenMotionLab/MotionGPT)  
  Additional Codebases:  
  - [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024)
  - Our codebase (simpler but not yet validated)

- **Project Code**  
  - **Dataloader**  
    - BEAT, Korean Motion Data <-- our codebase
  - **Tokenizing Model**  
    - VQ-VAE <-- MotionGPT
  - **Evaluation**  
    - Qualitative and Quantitative evaluation 
  - Tools: [BVHView](https://theorangeduck.com/page/bvhview), [Blender](https://www.blender.org/)

- **Implement the first version of motion tokenizer**
  - Reused our previous codebase for the data loader
  - Reused the model file from the MotionGPT codebase
  - Implemented an inference code for a single input BVH file
  - Displayed codebook indexes in a video

- **Retarget Korean Motion Data to BEAT skeleton structure using Blender Auto Rig Pro Addon**
  - Sample Data (5 BVH files)

- **Combine BEAT and Korean Motion Data**
  - preprocess all data from the two datasets
  - Trained a VQ-VAE model on the combined dataset

## Links/Resources

### References
- [Co-Speech Gesture Synthesis using Discrete Gesture Token Learning](https://arxiv.org/pdf/2303.12822)
- [BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis](https://arxiv.org/abs/2203.05297)
- [EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Expressive Masked Audio Gesture Modeling](https://arxiv.org/abs/2401.00374)
- [Large Motion Model for Unified Multi-Modal Motion Generation](https://arxiv.org/abs/2404.01284)

### Motion Retargeting
- **Auto Rig Pro (Blender plugin)**  
  [Auto Rig Pro on BlenderMarket](https://blendermarket.com/products/auto-rig-pro)  
- How to Use Auto-Rig Pro in Blender 4.2 to Create a Remap Preset .bmap File for Retargeting:
  [![Watch the tuto](https://img.youtube.com/vi/VqTWtiRrw5A/maxresdefault.jpg)](https://www.youtube.com/watch?v=VqTWtiRrw5A)

### Papers Using Retargeting
- **SAME: Skeleton-Agnostic Motion Embedding for Character Animation**  
  Section 4.3.1. discusses usage of MotionBuilder for retargeting.

## How to Use This Code

This code is based on the [Hydra Lightning Template](https://github.com/ashleve/lightning-hydra-template).

### Environment Setup

Follow these steps to set up the environment:

```bash
# Deactivate any active conda environment
conda deactivate

# Remove existing environment named MT_env (if any)
conda remove --name MT_env --all

# Create a new environment named MT_env with Python 3.8
conda create -n MT_env python=3.8

# Activate the new environment
conda activate MT_env

# Install PyTorch with CUDA support
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install iopath
conda install -c iopath iopath

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install remaining dependencies
pip install -r requirements.txt
```

### Code Workflow

#### 1. Preprocessing and remapin (or retargeting)

1. Update the paths in the Hydra config file `configs/paths/default.yaml` to point to the relevant datasets:
    - BEAT dataset `beat_data_dir`: [https://github.com/PantoMatrix/BEAT](https://github.com/PantoMatrix/BEAT)
    - Korean dataset `aihub_data_dir`: AIHub dataset

In addition to updating dataset paths, you must also configure the paths for the data preprocessing directories. 
  - `data_dir`: This is the base directory where all your data will be stored. Ensure that this directory has at least 6TB of free space available.
  - `pkl_path`: This path should be set relative to `data_dir`. It defines where the `.pkl` files will be saved. For example, if `data_dir` is set to `"E:/data/"`, then `pkl_path` could be `"E:/data/pkl/XXXXXXXXX.pkl"`. The `XXXXXXXXX` will be dynamically generated based on the BVH file's name during the run.
  - `out_bvh_path`: Similar to `pkl_path`, this path should also be set relative to `data_dir`. It specifies where the processed `.bvh` files will be saved. For example, `"E:/data/processed_bvh/XXXXXXXXX.bvh"`, where `XXXXXXXXX` will again be dynamically generated based on the BVH file's name.

2. Get the remap presets `.bmap` file for retargeting the armatures. [![Watch the tuto](https://img.youtube.com/vi/VqTWtiRrw5A/maxresdefault.jpg)](https://www.youtube.com/watch?v=VqTWtiRrw5A) and update the `remap_path` in the `convert_bvh.py`.
3. The BEAT dataset BVH files should be remapped using the `convert_bvh.py` script with Blender and the Auto Rig Pro addon.

4. Apply the `bvh_hips_translator.py` script to all the data to make sure hips (the root joint) are all at the same starting point (0,0,0).

5. Apply the `modify_bvh_frame_time.py` script to modify the frame rate of bvh files which need to.

6. Run the `prepare_data.py` script to set up the data.

7. Run `data_preprocess.py` to preprocess the data. Since all the data now follow the Korean AIHub skeleton structure, ensure that the dataset name in the config is set to `aihub` (under `configs/data`).

#### 2. Training the Tokenizer

- Run the `train.py` script to train the tokenizer on the preprocessed data. You can easily change parameters and hyperparameters under `configs/model`.

#### 3. Testing the Model

- To test the code on a new BVH file:
    1. Run `prepare_data_test.py`.
    2. Run `data_preprocess_test.py` using the path to the test BVH file in `configs/paths/data_dir_test`.

#### 4. Visualization

- You can visualize the results using:
    - [WandB](https://wandb.ai/)
    - [BVHView](https://theorangeduck.com/media/uploads/BVHView/bvhview.html)


### BVH File Processing Scripts

#### Overview
This repository contains a collection of Python scripts designed to process, modify, and manage BVH (Biovision Hierarchy) files. These scripts are useful for tasks such as frame time modification, file conversion, deletion of unwanted files, and more. Below is a description of each script and instructions on how to use them.

#### Scripts

#####  `find_bvh_with_most_frames.py`

**Description**:  
This script scans a directory to find the `.bvh` file with the most frames, specifically targeting files ending with `_translated.bvh`.

---

#####  `modify_bvh_frame_time.py`

**Description**:  
Modifies the frame time of `.bvh` files to match a new frame rate. The script can optionally rename the output files

---

##### `convert_bvh.py`

**Description**:  
This script is used for retargeting and converting `.bvh` files using Blender. It applies transformations to adjust skeletons and exports the files in a modified format.

**Usage**:  
- Imports and modifies `.bvh` files in Blender.

**Execution**:
```bash
blender --background --python convert_bvh.py
```

**Requirements**:
- Blender and Auto Rig Pro addon
- Python 3.x


---

##### `bvh_hips_translator.py`

**Description**:  
Normalizes the hips position in `.bvh` files, aligning all frames relative to the first hips position.

**Usage**:  
- Translates and normalizes `.bvh` files, ensuring consistent hips positioning throughout the animation.
