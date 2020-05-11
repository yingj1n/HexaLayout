# HexaLayout: From images to roadmap and object detection on BEV
authors: Ying Jin, Xiao Li, Shiqing Li

working repository for dl final project

### Abstract 

### Model architecture
<img src="model_architecture.png" width="600">

### Usage 
Our implementation does not rely on any pre-trained networks, rather it is all built from scratch. To fully utilize our model, please download dataset from [here](https://drive.google.com/drive/folders/1wXQmPyoEGkHB8Jq2uZ2uSZa7TgLhw3Mb).

**Train HexaLayout** <br>
run `python train_hexalayout.py --folder_dir ./data/root/dir/`

**Train HexaLayout with Discriminator** <br>
run `python train_hexalayout_discri.py --folder_dir ./data/root/dir/`

**Train HexaLayout with depth encoder** <br>
run `python train_hexalayout_w_de.py --folder_dir ./data/root/dir/ --depth_model_dir <depth model dir here>` 

For our trained depth models, check [this drive folder](https://drive.google.com/drive/folders/1xdHYG5ti_S0rtzNTagoOryZWziu02MQY?usp=sharing) 

For codes to train the Monodepth2 model, check [here](https://github.com/Heimine/monodepth2)

