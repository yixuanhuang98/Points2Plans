# Points2Plans: From Point Clouds to Long-Horizon Plans with Composable Relational Dynamics 
Code to accompany our paper: _Points2Plans: From Point Clouds to Long-Horizon Plans with Composable Relational Dynamics_. [[PDF]](arxiv.org/pdf/2408.14769) [[Website]](https://sites.google.com/stanford.edu/points2plans)

## Approach Overview
![Overview figure](images/teaser.png)

This repository includes: 

* :hammer_and_wrench: A relational dynamics model that excels at long-horizon prediction of point cloud states without the need to train on multi-step data
* :rocket: A latent-geometric space dynamics rollout strategy that significantly increases the horizons over which predicted point cloud states are reliable for planning  
* 🦾 A task planning and goal prediction module using Large Language Models (LLMs)

## Setup 
### System Requirements
This codebase is primarily tested on Ubuntu 20.04, an NVIDIA GeForce RTX 3090 Ti, and CUDA 11.7.  

### Virtual Env Installation
```
conda env create -f conda_env.yml
```

## Task Planning and Goal Prediction Module with LLMs
```bash
python LLM/scripts/llm_planner.py \ 
    --model-config LLM/configs/models/pretrained/generative/$Model \ 
    --prompt-config LLM/configs/prompts/evaluation/p1/$Task  \
    --api-key $YourAPIKey
```

## Relational Dynamics 
### Quick Start with Pretrained Models
- Download pretrained models from [this link](https://huggingface.co/datasets/ll4ma-lab/Points2Plans/tree/main/pretrained_model)
- Download test data for [constrained packing task](https://huggingface.co/datasets/ll4ma-lab/Points2Plans/tree/main/test)

```bash
python relational_dynamics/main.py \
    --result_dir $PretrainedModelDir  \
    --checkpoint_path $PretrainedModelDir/checkpoint/pretrained.pth \ 
    --test_dir $TestDataDir  \
    --test_max_size $TestSize 
```

### Training

- Download [training datasets](https://huggingface.co/datasets/ll4ma-lab/Points2Plans/tree/main/train)

To generate your own data, please refer to our simulation repository using [[isaacgym]](https://bitbucket.org/robot-learning/ll4ma_isaac/src/CoRL_2024/). 

```bash
python relational_dynamics/main.py \
    --result_dir $YourResultDir  \
    --train_dir $TrainingDataDir \
    --batch_size $BatchSize \
    --num_epochs $TrainingEpochs \
    --max_size $TrainingSize 
```

## Baseline: [eRDTransformer](https://sites.google.com/view/erelationaldynamics)

### Training 
```bash
python relational_dynamics/main.py \
    --result_dir $YourResultDir  \
    --train_dir $TrainingDataDir \
    --batch_size $BatchSize \
    --num_epochs $TrainingEpochs \
    --delta_forward False \ 
    --latent_forward True \ 
    --max_size $TrainingSize 
```

### Test 
- Download pretrained models from [this link](https://huggingface.co/datasets/ll4ma-lab/Points2Plans/tree/main/baseline_pretrained)

```bash
python relational_dynamics/main.py \
    --result_dir $PretrainedModelDir  \
    --checkpoint_path $PretrainedModelDir/checkpoint/baseline_pretrained.pth \ 
    --test_dir $TestDataDir  \
    --delta_forward False \ 
    --latent_forward True \ 
    --test_max_size $TestSize 
```


## Citation
If you find our work useful in your research, please cite:
```
@misc{huang-2024-points2plans,
author = {Yixuan Huang and Christopher Agia and Jimmy Wu and Tucker Hermans and Jeannette Bohg},
title = {{Points2Plans: From Point Clouds to Long-Horizon Plans with Composable Relational Dynamics}},
url = {sites.google.com/stanford.edu/points2plans},
year = 2024
}
```