# DexGraspMotionChallenge2025

## Overview

This repository provides example code for training and testing on grasping trajectories of a single object. It demonstrates how to set up the training pipeline and evaluate the performance of learned policies within a simulated environment.

## 1. Environment Setup

## 2. Dataset Download

## 3. Training and Testing Examples

We provide two training approaches based on different learning frameworks.Both methods utilize [DexRep](https://arxiv.org/pdf/2303.09806), a representation for dexterous grasping that encodes both geometric and spatial hand-object information. DexRep consists of three components: (1) Occupancy Feature, (2) Surface Feature, and (3) Local-Geo Feature.

**Method 1: MLP + DexRep**

In this baseline, we use a multi-layer perceptron (MLP) as the policy network trained on top of DexRep features extracted from hand-object configurations.

**Method 2: DP3 + DexRep**

This approach is inspired by [3D diffusion policy](https://arxiv.org/abs/2403.03954).

We integrate DexRep as the feature representation within the DP3 diffusion-based policy learning framework.

### Training Example

**Method 1**

Run the training with:

<pre><code>cd dexgrasp
python train_bc_lighting_dexrep.py</code></pre>

**Method 2**

Run the training with:

<pre><code>cd dexgrasp
python train_bc_lighting_dp3_dexrep.py</code></pre>

### Testing Example

**Method 1**

Run the following command to perform testing:

<pre><code>python -u bc_env_infer.py --task=ShadowHandGraspDexRepIjrr --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --headless</code></pre>

**Method 2**

Run the following command to perform testing:

<pre><code>python -u bc_env_infer_multisteps.py --task=ShadowHandGraspDexRepIjrr --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp --headless</code></pre>

