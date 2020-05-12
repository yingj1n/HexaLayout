# HexaLayout: From images to roadmap and object detection on BEV
Authors: Ying Jin, Xiao Li, Shiqing Li

### Abstract
In this work, we attempt to solve a novel problem of estimating the surrounding bird eye layout of a complex driving scenario with six different camera perspective images only. While existing literature aims to approach this set of problem with a pre-trained neural network with ability todetect object given images, and rely on intrinsic and extrinsic calibration data to transform from perspective view(s) to bird eye view (BEV), we followed the idea from MonoLayout (Mani et al., 2020) to derive a single model that transform from perspective view to BEV and segments objects all-in-one. We also propose an architecture using both ResNet (He et al., 2015) and U-Net (Ron-neberger et al., 2015) that outperforms the Mono-Layout architecture.

### Model architecture
<img src="model_architecture.png" width="600">

### Usage 
Our implementation does not rely on any pre-trained networks, rather it is all built from scratch. The original dataset can be downloaded from [here](https://drive.google.com/drive/folders/1wXQmPyoEGkHB8Jq2uZ2uSZa7TgLhw3Mb).

**Train HexaLayout** (Leaderboard submission.)<br>
run `python train_hexalayout.py --folder_dir /data/root/dir/`

**Train HexaLayout with Discriminator** <br>
run `python train_hexalayout_discri.py --folder_dir /data/root/dir/`

**Train HexaLayout with depth encoder** <br>
run `python train_hexalayout_w_depth.py --folder_dir /data/root/dir/ --depth_model_dir <depth model dir here>`

For our trained depth models, check [this drive folder](https://drive.google.com/drive/folders/1xdHYG5ti_S0rtzNTagoOryZWziu02MQY?usp=sharing)

Check our fork for training the Monodepth2 model [here](https://github.com/Heimine/monodepth2).

