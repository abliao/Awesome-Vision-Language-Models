# Awesome-Vision-Language-Models

## Framework

- 基础知识

  - 预训练方法

    - 对比学习
    - 生成学习
    - 对比和生成混合

  - 多模态融合

    视觉语言怎么统一。

    根据应用的需求，输入和输出会拥有更多模态，怎么统一多模态输入和输出。

  - 多模态大模型（LMM）

    受LLM启发，多模态大模型的研究也有了飞速进展

    - 指令微调
    - 上下文学习
    - 思维链推理

- Tasks

  - ？VQA
  - 分类（hrh）
  - 检测（hjh）
  - 分割（hjh）
  - ？生成
  - 视频

- 应用

  - 医疗
  - *文档大模型*（huawei）
  - 手机终端
  - 自动驾驶（huawei）
  - Embodied AI（zkd）
  - AI Agent（zkd）

## Papers

- [Introduction](#Introduction)
- [基础知识](#基础知识)
  - [模型构建框架](#模型构建框架)
- [Tasks](#tasks)
  - [检测](#检测)
  - [分割](#分割)
  - [VQA](#vqa)
  - [生成](#生成)
  - [视频](#视频)
- [Training](#training)
  - [数据集](#数据集)
  - [指令优化](#指令优化)
- [Application](#application)
  - [医疗](#医疗)
    - 研究现状
    - 数据集
    - 方法
    - 评估方法
    - 挑战和展望
  - [文档大模型](#文档大模型)
  - [自动驾驶](#自动驾驶)
  - 手机终端(#手机终端)
  - [Embodied AI](#embodied-ai)
  - [AI Agent](#ai-agent)

## Introduction
  介绍survey的必要性，和之前的survey的不同，强调应用介绍的重要性

## 基础知识 / Method
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**Vision-Language Models for Vision Tasks: A Survey**](https://arxiv.org/pdf/2304.00685.pdf)|-|[Code](https://github.com/jingyi0000/VLM_survey)|
|[**A Survey on Multimodal Large Language Models**](https://arxiv.org/pdf/2306.13549.pdf)|-|[Code](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)|
|[**Multimodal Foundation Models: From Specialists to General-Purpose Assistants**](https://arxiv.org/abs/2309.10020)|[Project Page](https://vlp-tutorial.github.io/2023/)|-|
### 预训练方法

| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|-|-|-|

### 多模态融合

### 多模态大模型

## Tasks

### 检测
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|-|-|-|

### 分割
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|-|-|-|

### VQA
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|-|-|-|

### 生成
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|-|-|-|

### 视频
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**SPOT! Revisiting Video-Language Models for Event Understanding**](https://arxiv.org/pdf/2311.12919v2.pdf)|-|-|
|[**LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models**](https://arxiv.org/pdf/2311.17043.pdf)||(Code)(https://github.com/dvlab-research/LLaMA-VID)|

## Application
### 医疗
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering**](https://arxiv.org/pdf/2305.10415.pdf)||[Code](https://github.com/xiaoman-zhang/PMC-VQA)|
|[**Med-Flamingo: a Multimodal Medical Few-shot Learner**](https://arxiv.org/abs/2307.15189)||[Code](https://github.com/snap-stanford/med-flamingo)|
|[**BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks**](https://arxiv.org/abs/2305.17100)||[Code](https://github.com/taokz/BiomedGPT/tree/main)|
|[**Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation**](https://arxiv.org/abs/2304.12620)||[Code](https://github.com/WuJunde/Medical-SAM-Adapter)|

### 文档大模型
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**](https://arxiv.org/abs/2204.08387)|||
|[**ERNIE-Layout: Layout Knowledge Enhanced Pre-training for Visually-rich Document Understanding**](https://arxiv.org/abs/2210.06155)||[Code](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout)|
|[**GeoLayoutLM: Geometric Pre-training for Visual Information Extraction**](https://arxiv.org/abs/2304.10759)||[Code](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM)|

### 自动驾驶
| Paper                                                        |                 Project WebSite                 |                        Code                         |
| ------------------------------------------------------------ | :---------------------------------------------: | :-------------------------------------------------: |
| [**DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving**](https://arxiv.org/pdf/2309.09777.pdf) | [Project Page](https://drivedreamer.github.io/) | [Code](https://github.com/JeffWang987/DriveDreamer) |
| [**GAIA-1: A Generative World Model for Autonomous Driving**](https://arxiv.org/abs/2309.17080) |                                                 |                                                     |
| [**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/abs/2312.00438) |  [Project Page](https://vlm-driver.github.io/)  |   [Code](https://github.com/vlm-driver/Dolphins)    |
| [**Vision Language Models in Autonomous Driving and Intelligent Transportation Systems**](https://arxiv.org/abs/2310.14414) |                                                 |                                                     |

### 手机终端

### Embodied AI

#### 规划

| Paper                                                        |                      Project WebSite                       |                             Code                             |
| ------------------------------------------------------------ | :--------------------------------------------------------: | :----------------------------------------------------------: |
| [**LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action**](https://arxiv.org/abs/2207.04429) |    [Project Page](https://sites.google.com/view/lmnav)     |       [Code](https://github.com/blazejosinski/lm_nav)        |
| [**LLM as A Robotic Brain: Unifying Egocentric Memory and Control**](https://arxiv.org/abs/2304.09349) |                                                            |                                                              |
| [**CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation**](https://arxiv.org/abs/2203.10421) |                                                            |                                                              |
| [**PaLM-E: An Embodied Multimodal Language Model**](https://palm-e.github.io/assets/palm-e.pdf) |         [Project Page](https://palm-e.github.io/)          |                                                              |
| [**Do As I Can, Not As I Say: Grounding Language in Robotic Affordances**](https://say-can.github.io/assets/palm_saycan.pdf) |         [Project Page](https://say-can.github.io/)         | [Code](https://github.com/google-research/google-research/tree/master/saycan) |
| [**Video Language Planning**](https://arxiv.org/abs/2310.10625) | [Project Page](https://video-language-planning.github.io/) | [Code](https://github.com/video-language-planning/vlp_code)  |

#### 执行

| Paper                                                        | Project WebSite                                          | Code                                                         |
| ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| [**VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models**](https://arxiv.org/abs/2307.05973) | [Project Page](https://voxposer.github.io/)              | [Code](https://github.com/huangwl18/VoxPoser)                |
| [**Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions**](https://arxiv.org/abs/2309.10150) | [Project Page](https://qtransformer.github.io/)          |                                                              |
| [**RoboAgent: Towards Sample Efficient Robot Manipulation with Semantic Augmentations and Action Chunking**](https://robopen.github.io/media/roboagent.pdf) | [Project Page](https://robopen.github.io/)               | [Code](https://github.com/robopen/roboagent/)                |
| [**RT-1: Robotics Transformer for Real-World Control at Scale**](https://arxiv.org/abs/2212.06817) | [Project Page](https://robotics-transformer1.github.io/) | [Code](https://github.com/google-research/robotics_transformer) |
| [**RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**](https://arxiv.org/abs/2307.15818) | [Project Page](https://robotics-transformer.github.io/)  |                                                              |
| [**Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?**](https://arxiv.org/abs/2303.18240) | [Project Page](https://eai-vc.github.io/)                | [Code](https://github.com/facebookresearch/eai-vc)           |
| [**On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline**](https://arxiv.org/abs/2212.05749) |                                                          | [Code](https://github.com/gemcollector/learning-from-scratch) |
| [**Open-World Object Manipulation using Pre-trained Vision-Language Models**](https://arxiv.org/abs/2303.00905) | [Prject Page](https://robot-moo.github.io/)              |                                                              |
| [**Large Language Models as Generalizable Policies for Embodied Tasks**](https://arxiv.org/abs/2310.17722) | [Project Page](https://llm-rl.github.io/)                |                                                              |

#### Generalist Model

| Paper                                                        | Project WebSite                                              | Code                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**An Embodied Generalist Agent in 3D World**](https://arxiv.org/abs/2311.12871) | [Project Page](https://embodied-generalist.github.io/)       | [Code](https://github.com/embodied-generalist/embodied-generalist) |
| [**3D-LLM: Injecting the 3D World into Large Language Models**](https://arxiv.org/abs/2307.12981) | [Project Page](https://vis-www.cs.umass.edu/3dllm/)          | [Code](https://github.com/UMass-Foundation-Model/3D-LLM)     |
| [**A Generalist Agent**](https://arxiv.org/abs/2205.06175)   | [Project Page](https://deepmind.google/discover/blog/a-generalist-agent/) |                                                              |
| [**PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training**](https://arxiv.org/abs/2209.11133) | [Project Page](https://www.microsoft.com/en-us/research/?post_type=msr-blog-post&p=868116) | [Code](https://github.com/microsoft/PACT)                    |
| [**EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought**](https://arxiv.org/abs/2305.15021) | [Project Page](https://embodiedgpt.github.io/)               | [Code](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch)   |

#### Benchmark

| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks**](https://arxiv.org/abs/2108.03272)|[Project Page](https://svl.stanford.edu/igibson/)|[Code](https://github.com/StanfordVL/iGibson)|
|[**BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments**](https://arxiv.org/abs/2108.03332)|[Project Page](https://behavior.stanford.edu/)|[Code](https://github.com/StanfordVL/behavior)|
|[**Habitat 2.0: Training Home Assistants to Rearrange their Habitat**]()|[Project Page](https://aihabitat.org/)|[Code](https://github.com/facebookresearch/habitat-lab/tree/v0.3.0)|
|[**Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots**](https://arxiv.org/abs/2310.13724)|[Project Page](https://aihabitat.org/)|[Code](https://github.com/facebookresearch/habitat-lab/tree/v0.3.0)|
|[**Transporter Networks: Rearranging the Visual World for Robotic Manipulation**](https://arxiv.org/abs/2010.14406)|[Project Page](https://transporternets.github.io/)|[Code](https://github.com/google-research/ravens)|
|[**robosuite: A Modular Simulation Framework and Benchmark for Robot Learning**](https://arxiv.org/abs/2009.12293)|[Project Page](https://robosuite.ai/)|[Code](https://github.com/ARISE-Initiative/robosuite)|
|[**HandoverSim: A Simulation Framework and Benchmark for Human-to-Robot Object Handovers**](https://arxiv.org/abs/2205.09747)|[Project Page](https://handover-sim.github.io/)|[Code](https://github.com/NVlabs/handover-sim)|
|[**ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations**](https://arxiv.org/abs/2107.14483)|[Project Page](https://sapien.ucsd.edu/challenges/maniskill/2021/)|[Code](https://github.com/haosulab/ManiSkill)|
|[**ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills**](https://arxiv.org/abs/2302.04659)|[Project Page](https://maniskill2.github.io/)|[Code](https://github.com/haosulab/ManiSkill2)|
|[**RLBench: The Robot Learning Benchmark & Learning Environment**](https://arxiv.org/abs/1909.12271)|[Project Page](https://sites.google.com/view/rlbench)|[Code](https://github.com/stepjam/RLBench)|
|[**ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks**](https://arxiv.org/abs/1912.01734)|[Project Page](https://askforalfred.com/)|[Code](https://github.com/askforalfred/alfred)|
|[**VIMA: General Robot Manipulation with Multimodal Prompts**](https://arxiv.org/abs/2210.03094)|[Project Page](https://vimalabs.github.io/)|[Code](https://github.com/vimalabs/VIMA)|
|[**VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation**](https://arxiv.org/abs/2206.08522)|[Project Page](https://sites.google.com/ucsc.edu/vlmbench/home)|[Code](https://github.com/eric-ai-lab/VLMbench)|
|[**HandMeThat: Human-Robot Communication in Physical and Social Environments**](https://openreview.net/pdf?id=nUTemM6v9sv)|[Project Page](https://sites.google.com/view/hand-me-that/)|[Code](https://github.com/Simon-Wan/HandMeThat-Release)|
|[**RM-PRT: Realistic Robotic Manipulation Simulator and Benchmark with Progressive Reasoning Tasks**](https://arxiv.org/abs/2306.11335)|[Project Page](https://necolizer.github.io/RM-PRT/)|[Code](https://github.com/Necolizer/RM-PRT)|
|[**MO-VLN: A Multi-Task Benchmark for Open-set Zero-Shot Vision-and-Language Navigation**](https://arxiv.org/abs/2306.10322)|[Project Page](https://mligg23.github.io/MO-VLN-Site/)|[Code](https://github.com/mligg23/MO-VLN/)|
|[**Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation**](https://peract.github.io/paper/peract_corl2022.pdf)|[Project Page](https://peract.github.io/)|[Code](https://github.com/peract/peract)|
|[**CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning**](https://arxiv.org/pdf/2010.04296.pdf)|[Project Page](https://sites.google.com/view/causal-world/home)|[Code](https://github.com/rr-learning/CausalWorld#sim2real)|
|[**Interactive Language: Talking to Robots in Real Time**](https://arxiv.org/abs/2210.06407)|[Project Page](https://interactive-language.github.io/)|[Code](https://github.com/google-research/language-table)|

### AI Agent

| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**MM-REACT**](https://arxiv.org/abs/2303.11381)|[Project Page](https://multimodal-react.github.io/)|[Code](https://github.com/microsoft/MM-REACT)|
|[**Visual ChatGPT**](https://arxiv.org/abs/2303.04671)||[Code](https://github.com/moymix/TaskMatrix)|
|[**CogAgent**](https://arxiv.org/abs/2312.08914)||[Code](https://github.com/THUDM/CogVLM)|
