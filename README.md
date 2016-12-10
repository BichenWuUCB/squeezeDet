## _SqueezeDet:_ Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving
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
  Let's call the top level directory of SqueezeDet as `$SQDT_ROOT`. 

## Demo:
- Download SqueezeDet model parameters from [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0), untar it, and put it under `$SQDT_ROOT/data/` If you are using command line, type:

  ```Shell
  cd $SQDT_ROOT/data/
  wget https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz
  tar -xzvf model_checkpoints.tgz
  rm model_checkpoints.tgz
  ```


- Now we can run the demo. To detect the sample image `$SQDT_ROOT/data/sample.png`,

  ```Shell
  cd $SQDT_ROOT/data/
  python ./src/demo.py
  ```
  If the installation is correct, the detector should generate this image: ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/out_sample.png)

  To detect other image(s), use the flag `--input_path=./data/*.png` to point to input image(s). Input image(s) will be scaled to the resolution of 1242x375 (KITTI image resolution), so it works best when original resolution is close to that.  

- SqueezeDet is a real-time object detector, which can be used to detect videos. The video demo will be released later.

## Training/Validation:
- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

- Now we need to split the training data into a training set and a vlidation set. 

  ```Shell
  cd $SQDT_ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
  `trainval.txt` contains indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. 

  When above two steps are finished, the structure of `$SQDT_ROOT/data/KITTI/` should at least contain:

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```

- Next, download the CNN model pretrained for ImageNet classification:
  ```Shell
  cd $SQDT_ROOT/data/
  # SqueezeNet
  wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
  tar -xzvf SqueezeNet.tgz
  # ResNet50 
  wget https://www.dropbox.com/s/p65lktictdq011t/ResNet.tgz
  tar -xzvf ResNet.tgz
  # VGG16
  wget https://www.dropbox.com/s/zxd72nj012lzrlf/VGG16.tgz
  tar -xzvf VGG16.tgz
  ```

- Now we can start training. Training script can be found in `$SQDT_ROOT/scripts/train.sh`, which contains commands to train 4 models: SqueezeDet, SqueezeDet+, VGG16+ConvDet, ResNet50+ConvDet. Un-comment the model you want to train, and then, type:

  ```Shell
  cd $SQDT_ROOT/data/
  ./scripts/train.sh
  ```

  Training logs are saved to the directory specified by `--train_dir`. 

- At the same time, you can launch evaluation by 

  ```Shell
  cd $SQDT_ROOT/data/
  ./scripts/eval_train.sh
  ./scripts/eval_val.sh
  ```

  If you've changed the `--train_dir` in the training script, make sure to also change `--checkpoint_dir` in the evaluation script to the same as `--train_dir` so evaluation script knows where to find the checkpoint. Also, it's recommended to put `--train_dir` and  `--eval_dir` under the same `$LOG_DIR` such that tensorboard can load both training and evaluation logs. 

  The two scripts simultaneously evaluate the model on training and validation set. The training script keeps dumping checkpoint (model parameters) to the training directory once every 1000 steps (step size can be changed). Once a new checkpoint is saved, evaluation threads load the new checkpoint file and evaluate them on training and validation set. 

- Finally, to monitor training and evaluation process, you can use tensorboard by

  ```Shell
  tensorboard --logdir=$LOG_DIR
  ```
  From tensorboard, you should be able to see a lot of information including loss, Average Precisions, error analysis, example detections, model visualization, etc.

  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/detection_analysis.png)
  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/graph.png)
  ![alt text](https://github.com/BichenWuUCB/squeezeDet/blob/master/README/det_img.png)
