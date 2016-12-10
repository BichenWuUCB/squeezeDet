## _SqueezeDet_: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving
By Bichen Wu, Forrest Iandola, Peter H. Jin, Kurt Keutzer (UC Berkeley & DeepScale)

This repository contains a tensorflow implementation of SqueezeDet, a convolutional neural network based object detector described in our paper: https://arxiv.org/abs/1612.01051. If you find this work useful for your research, please consider citing:

    @inproceedings{squeezedet,
        Author = {Bichen Wu and Forrest Iandola and Peter H. Jin and Kurt Keutzer},
        Title = {SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving},
        Journal = {arXiv:1612.01051},
        Year = {2016}
    }
    
## Installation:
- Prerequisites:
    - Follow instructions to install Tensorflow: https://www.tensorflow.org.
    - Install opencv: http://opencv.org
    - Other packages that you might also need: easydict, joblib. You can use pip to install these packages:
    
    ```Shell
    pip install easydict
    pip install joblib
    ```
- Clone the SqueezeDet repository:

  ```Shell
  git clone https://github.com/BichenWuUCB/squeezeDet.git
  ```
  Let's call the top level directory of SqueezeDet as `SQ_ROOT`. 
- Download SqueezeDet model parameters from [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0), untar it, and put it under `SQ_ROOT/data/` If you are using command line, type:

```Shell
cd $SQ_ROOT/data/
wget https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz
tar -xzvf model_checkpoints.tgz
rm model_checkpoints.tgz
```

- Now we can run some demos. To detect the sample image given in `SQ_ROOT/data`,

```Shell
cd $SQ_ROOT/data/
python ./src/demo.py
```
If the installation is correct, the detector should generate this image:

To detect multiple images, use the flag `--input_path=./data/*.png` to specify images. Input images will be scaled to the resolution of 1242x375 (KITTI image resolution), so it works best when input images' original resolution is close to that.  




