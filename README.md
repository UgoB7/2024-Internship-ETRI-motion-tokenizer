# Motion Token Learning

## Authors
- Youngwoo
- Ugo (intern)


## Overview
The goal of this project is to train a tokenizer for motion data, leveraging multiple motion datasets to create a more general tokenizer. This internship involves exploring existing datasets and tokenizing methods.

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

- **Retarget Korean Motion Data to BEAT skeleton structure**
  - Sample Data (5 BVH files)

- **Combine BEAT and Korean Motion Data**
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
  [Tutorial Video](https://www.youtube.com/watch?v=HHnt-3uLSUo)
  
- **Scripts**  
  - Found script on GitHub (not yet tested): [Render Script](https://github.com/zjp-shadow/CharacterGen/blob/6fda5658a3d322ed75b913b93e61aa2d6c08db03/render_script/blender/render.py#L42)

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

#### 1. Preprocessing Raw BVH Files

1. Update the paths in the Hydra config file `configs/paths/default.yaml` to point to the relevant datasets:
    - BEAT dataset: [https://github.com/PantoMatrix/BEAT](https://github.com/PantoMatrix/BEAT)
    - Korean dataset: AIHub dataset

2. Run the `prepare_data.py` script to set up the data.

3. Run `data_preprocess.py` to preprocess the data. Since all the data now follow the Korean AIHub skeleton structure, ensure that the dataset name in the config is set to `aihub` (under `configs/data`). The BEAT dataset BVH files were remapped using the `convert_bvh.py` script with Blender and the Auto Rig Pro addon.

#### 2. Training the Tokenizer

- Run the `train.py` script to train the tokenizer on the preprocessed data.

#### 3. Testing the Model

- To test the code on a new BVH file:
    1. Run `prepare_data_test.py`.
    2. Run `data_preprocess_test.py` using the path to the test BVH file in `configs/paths/data_dir_test`.

#### 4. Visualization

- You can visualize the results using:
    - [WandB](https://wandb.ai/)
    - [BVHView](https://theorangeduck.com/media/uploads/BVHView/bvhview.html)