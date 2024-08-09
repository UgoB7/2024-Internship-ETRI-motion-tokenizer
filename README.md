# Motion Token Learning

## Authors
- Youngwoo
- Ugo


## Overview
The goal of this project is to train a tokenizer for motion data, leveraging multiple motion datasets to create a more general and robust tokenizer. This involves exploring existing datasets and tokenizing methods.

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
