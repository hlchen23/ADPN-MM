# ADPN: Adaptive Dual-branch Promoted Network

Repo for **ACM 23'MM** accepted paper "*Curriculum-Listener: Consistency- and Complementarity-Aware Audio-Enhanced Temporal Sentence Grounding*". This paper proposes solutions for the Temporal Sentence Grounding task from an audio-visual collaborative perspective.

Houlun Chen, Xin Wang\*, Xiaohan Lan, Hong Chen, Xuguang Duan, Jia Jia\*, Wenwu Zhu\*.

\* Corresponding authors.

![framework](figures/framework.png)

## Abstract

Temporal Sentence Grounding aims to retrieve a video moment given a natural language query. Most existing literature merely focuses on visual information in videos without considering the naturally accompanied audio which may contain rich semantics. The few works considering audio simply regard it as an additional modality, overlooking that: i) it's non-trivial to explore consistency and complementarity between audio and visual; ii) such exploration requires handling different levels of information densities and noises in the two modalities. To tackle these challenges, we propose **A**daptive **D**ual-branch **P**romoted **N**etwork (ADPN) to exploit such consistency and complementarity: i) we introduce a dual-branch pipeline capable of jointly training visual-only and audio-visual branches to simultaneously eliminate inter-modal interference; ii) we design **T**ext-**G**uided **C**lues **M**iner (TGCM) to discover crucial locating clues via considering both consistency and complementarity during audio-visual interaction guided by text semantics; iii) we propose a novel curriculum-based denoising optimization strategy, where we adaptively evaluate sample difficulty as a measure of noise intensity in a self-aware fashion. Extensive experiments show the state-of-the-art performance of our method.



## Installation

1. Clone the repository

```bash
git clone https://github.com/hlchen23/ADPN-MM.git
```

2. Set up the environment

Use Anaconda and easily build up the required environment by

```bash
cd ADPN-MM
conda env create -f env.yml
```

3. Data Preparation

We use GloVe-840B-300d for text embeddings, I3D visual features and PANNs audio features for Charades-STA dataset, and C3D visual features and VGGish audio features for ActivityNet Captions dataset. Download [data](https://pan.baidu.com/s/1LxdASuOzueq_4YpEr2muAA?pwd=5w4h), touch `ADPN-MM/data`, and ensure the following directory structure.

```
|--data
|  |--dataset
|     |--activitynet
|     |     |--train.json
|     |     |--val_1.json
|     |     |--val_2.json
|     |--charades
|     |     |--charades_sta_test.txt
|     |     |--charades_sta_train.txt
|     |     |--charades.json
|  |--features
|     |--activitynet
|     |     |--audio
|     |     |     |--VGGish.pickle
|     |     |--c3d_video
|     |     |     |--feature_shapes.json
|     |     |     |--v___c8enCfzqw.npy
|     |     |     |--...(*.npy)
|     |--charades
|     |     |--audio
|     |     |     |--0A8CF.npy
|     |     |     |--...(*.npy)
|     |     |--i3d_video
|     |     |     |--feature_shapes.json
|     |     |     |--0A8CF.npy
|     |     |     |--...(*.npy)
```



4. Training

We provide bash scripts to train our model, `train_ch.sh` and `train_anet.sh`. For example,

```bash
python main.py --task <charades|activitynet> --mode train --gpu_idx <GPU INDEX>
```

5. Inference

We provide bash scripts to execute the inference process, `test_ch.sh` and `test_anet.sh`. For example,

```bash
python main.py --task <charades|activitynet> --mode test --gpu_idx <GPU INDEX> --test_path checkpoint/<charades|activitynet>
```

put the `config.json` and `<charades|activitynet>_ckpt.t7` files under the `--test_path`.



Check more information from `args` for training and test in `main.py`.



## Acknowledgement

We follow the repo [VSLNet](https://github.com/26hzhang/VSLNet) for the code-running framework to quickly implement our work. We appreciate this great job.

## Cite

If you feel this repo is helpful to your research, please cite our work.

```
@inproceedings{10.1145/3581783.3612504,
    author = {Chen, Houlun and Wang, Xin and Lan, Xiaohan and Chen, Hong and Duan, Xuguang and Jia, Jia and Zhu, Wenwu},
    title = {Curriculum-Listener: Consistency- and Complementarity-Aware Audio-Enhanced Temporal Sentence Grounding},
    year = {2023},
    publisher = {Association for Computing Machinery},
    doi = {10.1145/3581783.3612504},
    booktitle = {Proceedings of the 31th ACM International Conference on Multimedia},
    pages = {}
}
```
