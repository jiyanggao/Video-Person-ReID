# Video-Person-ReID

This is the code repository for our tech report "Revisiting Temporal Modeling for Video-based Person ReID": https://arxiv.org/abs/1805.02104.
If you find this help your research, please cite

    @article{gao2018revisiting,
      title={Revisiting Temporal Modeling for Video-based Person ReID},
      author={Gao, Jiyang and Nevatia, Ram},
      journal={arXiv preprint arXiv:1805.02104},
      year={2018}
    }

### Introduction
This repository contains PyTorch implementations of temporal modeling methods for video-based person reID. It is forked from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).. Based on that, I implement (1) video sampling strategy for training and testing, (2) temporal modeling methods including temporal pooling, temporal attention, RNN and 3D conv. The base loss function and basic training framework remain the same as [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). **PyTorch 0.3.1, Torchvision 0.2.0 and Python 2.7** is used.

### Motivation
Although previous work proposed many temporal modeling methods and did extensive experiments, but it's still hard for us to have an "apple-to-apple" comparison across these methods. As the image-level feature extractor and loss function are not the same, which have large impact on the final performance. Thus, we want to test the representative methods under an uniform framework.

### Dataset
All experiments are done on MARS, as it is the largest dataset available to date for video-based person reID. Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
5. Use `-d mars` when running the training code.

### Usage
To train the model, please run

    python main_video_person_reid.py --arch=resnet50tp
arch could be resnet50tp (Temporal Pooling), resnet50ta (Temporal Attention), resnet50rnn (RNN), resnet503d (3D conv). For 3D conv, I use the design and implementation from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch), just minor modification is done to fit the network into this person reID system.

In my experiments, I found that learning rate has a significant impact on the final performance. Here are the learning rates I used (may not be the best): 0.0003 for temporal pooling, 0.0003 for temporal attention, 0.0001 for RNN, 0.0001 for 3D conv.

Other detailed settings for different temporal modeling could be found in `models/ResNet.py`

### Performance

| Model            | mAP |CMC-1 | CMC-5 | CMC-10 | CMC-20 |
| :--------------- | ----------: | ----------: | ----------: | ----------: | ----------: | 
| image-based      |   74.1  | 81.3 | 92.6 | 94.8 | 96.7 |
| pooling    |   75.8  | 83.1 | 92.8 | 95.3 | 96.8   |
| attention    |  76.7 | 83.3 | 93.8 | 96.0 | 97.4 |
| rnn    |   73.9 | 81.6 | 92.8 | 94.7 | 96.3 |
| 3d conv    |  70.5 | 78.5 | 90.9 | 93.9 | 95.9 |
