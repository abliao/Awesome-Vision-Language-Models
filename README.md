# Awesome-Multimodal-Foundation-Models And Applications

## Framework

- 基础知识

  - 多模态预训练

    - 视觉语言
      - 对比学习
      - 生成学习
      - 对比和生成混合
    - 多模态融合

  - 多模态大模型（LMM）

    受LLM启发，多模态大模型的研究也有了飞速进展

    - 结构
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
  - [手机终端](#手机终端)
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
|Image-Text|:-------------:|:------------:|
|[**Learning Transferable Visual Models From Natural Language Supervision**](https://arxiv.org/abs/2103.00020)|-|[Code](https://github.com/openai/CLIP)|
|[**Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision**](https://arxiv.org/abs/2102.05918)|-|[Code]|
|[**BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**](https://arxiv.org/abs/2201.12086)|-|[Code](https://github.com/salesforce/BLIP)|
|[**ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks**](https://arxiv.org/abs/1908.02265)|-|[Code](https://github.com/jiasenlu/vilbert_beta)|
|[**LXMERT: Learning Cross-Modality Encoder Representations from Transformers**](https://arxiv.org/abs/1908.07490)|-|[Code](https://github.com/airsplay/lxmert)|
|[**Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm**](https://arxiv.org/pdf/2110.05208.pdf)|-|[Code](https://github.com/Sense-GVT/DeCLIP)|
|[**FILIP: Fine-grained Interactive Language-Image Pre-Training**](https://arxiv.org/pdf/2111.07783.pdf)|-|[Code]|
|[**VL-BERT: Pre-training of Generic Visual-Linguistic Representations**](https://arxiv.org/abs/1908.03557)|-|[Code](https://github.com/jackroos/VL-BERT)|
|[**VisualBERT: A Simple and Performant Baseline for Vision and Language**](https://arxiv.org/abs/1908.07490)|-|[Code](https://github.com/uclanlp/visualbert)|
|[**Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training**](https://arxiv.org/abs/1908.06066)|-|[Code](https://github.com/microsoft/Unicoder)|
|[**Unified Vision-Language Pre-Training for Image Captioning and VQA**](https://arxiv.org/pdf/1909.11059.pdf)|-|[Code](https://github.com/LuoweiZhou/VLP)|
|[**UNITER: Learning Universal Image-text Representations**](https://arxiv.org/abs/1909.11740)|-|[Code](https://github.com/ChenRocks/UNITER)|
|[**DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting**](https://arxiv.org/pdf/2112.01518.pdf)|-|[Code](https://github.com/raoyongming/DenseCLIP)|
|[**Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks**](https://arxiv.org/abs/1912.03063)|-|[Code](https://github.com/LuoweiZhou/VLP)|
|[**Unified Vision-Language Pre-Training for Image Captioning and VQA**](https://arxiv.org/pdf/1909.11059.pdf)|-|[Code]|
|[**InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining**](https://arxiv.org/abs/2003.13198)|-|[Code](https://github.com/black4321/InterBERT)|
|[**Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks**](https://arxiv.org/pdf/2004.06165.pdf)|-|[Code](https://github.com/microsoft/Oscar)|
|[**Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers**](https://arxiv.org/abs/2004.00849)|-|[Code](https://github.com/microsoft/xpretrain)|
|[**ERNIE-VIL: KNOWLEDGE ENHANCED VISION-LANGUAGE REPRESENTATIONS THROUGH SCENE GRAPH**](https://arxiv.org/abs/2006.16934)|-|[Code]|
|[**DeVLBert: Learning Deconfounded Visio-Linguistic Representations**](https://arxiv.org/abs/2008.06884)|-|[Code](https://github.com/shengyuzhang/DeVLBert)|
|[**ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**](https://arxiv.org/pdf/2102.03334.pdf)|-|[Code](https://github.com/dandelin/ViLT)|
|[**UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning**](https://arxiv.org/abs/2012.15409)|-|[Code](https://github.com/PaddlePaddle/Research/tree/master/NLP/UNIMO)|
|[**VinVL: Revisiting Visual Representations in Vision-Language Models**](https://arxiv.org/abs/2101.00529)|-|[Code](https://github.com/pzzhang/VinVL)|
|[**Kaleido-BERT: Vision-Language Pre-training on Fashion Domain**](https://arxiv.org/abs/2103.16110)|-|[Code](https://github.com/mczhuge/Kaleido-BERT)|
|[**Product1M: Towards Weakly Supervised Instance-Level Product Retrieval via Cross-Modal Pretraining**](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhan_Product1M_Towards_Weakly_Supervised_Instance-Level_Product_Retrieval_via_Cross-Modal_Pretraining_ICCV_2021_paper.pdf)|-|[Code](https://github.com/zhanxlin/Product1M)|
|[**Align before Fuse: Vision and Language Representation Learning with Momentum Distillation**](https://arxiv.org/abs/2107.07651)|-|[Code](https://github.com/salesforce/ALBEF)|
|[**Florence: A New Foundation Model for Computer Vision**](https://arxiv.org/pdf/2111.11432.pdf)|-|[Code]|
|[**CoCa: Contrastive Captioners are Image-Text Foundation Models**](https://arxiv.org/pdf/2205.01917.pdf)|-|[Code](https://github.com/lucidrains/CoCa-pytorch)|




| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|audio-Text|:-------------:|:------------:|


| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|Video-Text|:-------------:|:------------:|
|[**VideoBERT: A Joint Model for Video and Language Representation Learning**](https://arxiv.org/abs/1904.01766)|-|[Code](https://github.com/ammesatyajit/VideoBERT)|
|[**Learning Video Representations Using Contrastive Bidirectional Transformers**](https://arxiv.org/abs/1906.05743)|-|[Code]|
|[**M-BERT: Injecting Multimodal Information in the BERT Structure**](https://arxiv.org/abs/1908.05787)|-|[Code]|
|[**BERT for Large-scale Video Segment Classification with Test-time Augmentation**](https://arxiv.org/abs/1912.01127)|-|[Code](https://github.com/hughshaoqz/3rd-Youtube8M-TM)|
|[**UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation**](https://arxiv.org/abs/2002.06353)|-|[Code](https://github.com/microsoft/UniVL)|
|[**ActBERT: Learning Global-Local Video-Text Representations**](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.html)|-|[Code](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/model_zoo/multimodal/actbert.md)|
|[**HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training**](https://arxiv.org/abs/2005.00200)|-|[Code](https://github.com/linjieli222/HERO)|
|[**Video-Grounded Dialogues with Pretrained Generation Language Models**](https://arxiv.org/abs/2006.15319)|-|[Code]|
|[**Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling**](https://arxiv.org/pdf/2102.06183.pdf)|-|[Code](https://github.com/jayleicn/ClipBERT)|
|[**CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval and captioning**](https://www.researchgate.net/publication/350992434_CLIP4Clip_An_Empirical_Study_of_CLIP_for_End_to_End_Video_Clip_Retrieval)|-|[Code](https://github.com/ArrowLuo/CLIP4Clip)|

### 多模态融合
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**Self-Supervised MultiModal Versatile Networks**](https://arxiv.org/pdf/2006.16228.pdf)|-|[Code](https://github.com/deepmind/deepmind-research/tree/master/mmv)|
|[**VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text**](https://arxiv.org/pdf/2104.11178.pdf)|-|[Code](https://github.com/google-research/google-research/tree/master/vatt)|
|[**Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videoss**](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Clustering_Networks_for_Self-Supervised_Learning_From_Unlabeled_Videos_ICCV_2021_paper.pdf)|-|[Code](https://github.com/brian7685/Multimodal-Clustering-Network)|
|[**OPT: Omni-Perception Pre-Trainer for Cross-Modal Understanding and Generation**](https://arxiv.org/pdf/2107.00249.pdf)|-|[Code]|
|[**M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining**](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_M5Product_Self-Harmonized_Contrastive_Learning_for_E-Commercial_Multi-Modal_Pretraining_CVPR_2022_paper.pdf)|-|[Code](https://xiaodongsuper.github.io/M5Product_dataset/)|
|[**OMNIVORE: A Single Model for Many Visual Modalities**](https://openaccess.thecvf.com/content/CVPR2022/papers/Girdhar_Omnivore_A_Single_Model_for_Many_Visual_Modalities_CVPR_2022_paper.pdf)|-|[Code](https://facebookresearch.github.io/omnivore)|
|[**MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound**](https://openaccess.thecvf.com/content/CVPR2022/papers/Zellers_MERLOT_Reserve_Neural_Script_Knowledge_Through_Vision_and_Language_and_CVPR_2022_paper.pdf)|-|[Code](https://rowanzellers.com/merlotreserve)|
|[**i-Code: An Integrative and Composable Multimodal Learning Framework**](https://arxiv.org/abs/2205.01818)|-|[Code](https://github.com/microsoft/i-Code)|
|[**One Model, Multiple Modalities: A Sparsely Activated Approach for Text, Sound, Image, Video and Code**](https://arxiv.org/pdf/2205.06126.pdf)|-|[Code](https://github.com/Yutong-Zhou-cv/Awesome-Multimodality)|
|[**ImageBind: One Embedding Space To Bind Them All**](https://arxiv.org/pdf/2305.05665.pdf)|-|[Code](https://facebookresearch.github.io/ImageBind)|



### 多模态大模型
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**](https://arxiv.org/abs/2304.10592)|[Project Page](https://minigpt-4.github.io/)|[Code](https://github.com/Vision-CAIR/MiniGPT-4)|
|[**Visual Instruction Tuning**](https://arxiv.org/abs/2304.08485)|-|[Code]()|
|[**CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks**](https://arxiv.org/pdf/2304.05653.pdf)|-|[Code](https://github.com/xmed-lab/CLIP_Surgery)|
|[**MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning**](https://arxiv.org/pdf/2310.09478.pdf)|-|[Code](https://github.com/Vision-CAIR/MiniGPT-4)|
|[**Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic**](https://arxiv.org/abs/2306.15195)|-|[Code](https://github.com/shikras/shikra)|
|[**BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**](https://arxiv.org/pdf/2301.12597.pdf)|-|[Code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)|
|[**VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks**](https://arxiv.org/pdf/2305.11175.pdf)|-|[Code]([https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/OpenGVLab/VisionLLM))|
|[**Kosmos-2: Grounding Multimodal Large Language Models to the World**](https://arxiv.org/pdf/2306.14824.pdf)|-|[Code](https://github.com/microsoft/unilm/tree/master/kosmos-2)|
|[**Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**](https://arxiv.org/pdf/2308.12966.pdf)|-|[Code](https://github.com/QwenLM/Qwen-VL)|
|[**mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality**](https://arxiv.org/pdf/2304.14178.pdf)|-|[Code](https://github.com/X-PLUG/mPLUG-Owl)|
|[**mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration**](https://arxiv.org/pdf/2311.04257.pdf)|-|[Code](https://www.zhqiang.org/mplug-owl2/)|
|[**InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**](https://arxiv.org/pdf/2305.06500.pdf)|-|[Code](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip)|
|[**VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning**](https://arxiv.org/abs/2102.10407)|-|[Code](https://github.com/Vision-CAIR/VisualGPT)|





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
|[**SPOT! Revisiting Video-Language Models for Event Understanding**](https://arxiv.org/pdf/2311.12919v2.pdf)|||
|[**LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models**](https://arxiv.org/pdf/2311.17043.pdf)||[Code](https://github.com/dvlab-research/LLaMA-VID)|
|[**MovieChat: From Dense Token to Sparse Memory for Long Video Understanding**](https://arxiv.org/pdf/2307.16449.pdf)|[Web](https://rese1f.github.io/MovieChat)|[Code](https://github.com/rese1f/MovieChat)|
|[**MIST : Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering**](https://arxiv.org/pdf/2212.09522.pdf)||[Code](https://github.com/showlab/)|
|[**Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models**](https://arxiv.org/pdf/2306.05424.pdf)||[Code](https://github.com/mbzuai-oryx/Video-ChatGPT)|
|[**RETRIEVAL-BASED VIDEO LANGUAGE MODEL FOR EFFICIENT LONG VIDEO QUESTION ANSWERING**](https://arxiv.org/pdf/2312.04931.pdf)|||
|[**VTimeLLM: Empower LLM to Grasp Video Moments**](https://arxiv.org/pdf/2311.18445.pdf)||[code](https://github.com/huangb23/VTimeLLM)|
|[**Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations**](https://arxiv.org/pdf/2211.11427.pdf)||[Code](https://github.com/jpthu17/EMCL)|
|[**EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding**](https://arxiv.org/pdf/2308.09126.pdf)|[Web](https://egoschema.github.io/)||
|[**MVBench: A Comprehensive Multi-modal Video Understanding Benchmark**](https://arxiv.org/pdf/2311.17005.pdf)||[Code](https://github.com/OpenGVLab/Ask-Anything)|
|[**Text-Conditioned Resampler For Long Form Video Understanding**](https://arxiv.org/pdf/2312.11897.pdf)|||
|[**Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding**](https://arxiv.org/pdf/2311.08046.pdf)||[Code](https://github.com/PKU-YuanGroup/Chat-UniVi)|
## Application
### 医疗
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering**](https://arxiv.org/pdf/2305.10415.pdf)||[Code](https://github.com/xiaoman-zhang/PMC-VQA)|
|[**Med-Flamingo: a Multimodal Medical Few-shot Learner**](https://arxiv.org/abs/2307.15189)||[Code](https://github.com/snap-stanford/med-flamingo)|
|[**BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks**](https://arxiv.org/abs/2305.17100)||[Code](https://github.com/taokz/BiomedGPT/tree/main)|
|[**Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation**](https://arxiv.org/abs/2304.12620)||[Code](https://github.com/WuJunde/Medical-SAM-Adapter)|
|A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics|||

### 文档大模型
| Paper                                             |  Project WebSite | Code |
|---------------------------------------------------|:-------------:|:------------:|
|[**LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking**](https://arxiv.org/abs/2204.08387)|||
|[**ERNIE-Layout: Layout Knowledge Enhanced Pre-training for Visually-rich Document Understanding**](https://arxiv.org/abs/2210.06155)||[Code](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout)|
|[**GeoLayoutLM: Geometric Pre-training for Visual Information Extraction**](https://arxiv.org/abs/2304.10759)||[Code](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM)|

### 自动驾驶
| Paper                                                        |                       Project WebSite                        |                        Code                         |
| ------------------------------------------------------------ | :----------------------------------------------------------: | :-------------------------------------------------: |
| [**DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving**](https://arxiv.org/pdf/2309.09777.pdf) |       [Project Page](https://drivedreamer.github.io/)        | [Code](https://github.com/JeffWang987/DriveDreamer) |
| [**GAIA-1: A Generative World Model for Autonomous Driving**](https://arxiv.org/abs/2309.17080) |                                                              |                                                     |
| [**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/abs/2312.00438) |        [Project Page](https://vlm-driver.github.io/)         |   [Code](https://github.com/vlm-driver/Dolphins)    |
| [**Vision Language Models in Autonomous Driving and Intelligent Transportation Systems**](https://arxiv.org/abs/2310.14414) |                                                              |                                                     |
| [**A Survey on Multimodal Large Language Models for Autonomous Driving**](https://arxiv.org/abs/2311.12320) | [Project Page](https://github.com/IrohXu/Awesome-Multimodal-LLM-Autonomous-Driving) |                                                     |

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
