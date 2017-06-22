The code has been tested on Ubuntu 14.04 and the only dependencies are from python which could be easily resolved with `pip`.

## Dependencies
1. Install CUDA-8.0 and CUDNN6 from NVIDIA.
2. Install all python dependencies: cv2, h5py, numpy, PIL, pytorch.
 
```
sudo apt-get install python-opencv, python-pip
sudo pip install h5py, numpy, Pillow
sudo pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
sudo pip install torchvision
```

## Model and Data
1. Download and uncompress the data from the competition. Put `IsoGD_phase_1` and `IsoGD_phase_2` containing original `.avi` videos in [dataset](dataset).
 This videos in these folders should follow a format (as provided by the competition):
```
dataset/IsoGD_phase_1/train/001/K_00001.avi
dataset/IsoGD_phase_1/train/001/M_00001.avi
...
dataset/IsoGD_phase_1/valid/001/K_00001.avi
dataset/IsoGD_phase_1/valid/001/M_00001.avi
...
dataset/IsoGD_phase_2/test/001/K_00001.avi
dataset/IsoGD_phase_2/test/001/M_00001.avi
```

2. Download and decompress pre-trained model `model_data.tar.gz` from 
[rgbd model and preprocessed pose data](https://www.dropbox.com/s/plzuw7coomwtkas/model_data.tar.gz?dl=0).
```
wget https://www.dropbox.com/s/plzuw7coomwtkas/model_data.tar.gz?dl=1 -O model_data.tar.gz
tar xf model_data.tar.gz
```
After decompression, you will find model and data in two new folders: [model](model) and [pose_h5](pose_h5) in the root folder of the code.

After all these steps, the code directory should have:
```
dataset/IsoGD_phase_1/train
dataset/IsoGD_phase_1/valid
dataset/IsoGD_phase_1/info
logs/test
model/*.model
pose_h5/phase1/*
pose_h5/phase2/*
isogd.py
main_rgbd_fusion.py
main_train_rgbd_c3d.py
rgbd_c3d.py
utils.py
README.md
```
And you are ready to do training or testing.

## Testing
We run our model by ourselves and prediction results used for submission can be found in [pred](logs/test/pred).

If you want to replicate these results with the pre-trained model, follow:
 
1. To evaluate on validation split (phase1) with RGB modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_rgb_init_depth.model \
--gpu_id 0,1,2,3 --evaluate 1 --eval_split val --modality focus_rgb 
```
2. To evaluate on validation split (phase1) with depth modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_depth_init_rgb.model \
--gpu_id 0,1,2,3 --evaluate 1 --eval_split val --modality focus_depth 
```
3. To evaluate on test split (phase2) with RGB modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_rgb_init_depth.model \
--gpu_id 0,1,2,3 --evaluate 1 --eval_split test --modality focus_rgb 
```
4. To evaluate on test split (phase2) with depth modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_depth_init_rgb.model \
--gpu_id 0,1,2,3 --evaluate 1 --eval_split test --modality focus_depth 
```

After this, [logs/test/score](logs/test/score) folder will be populated with original prediction score for each valid/test sample.

These score will be used for late fusion and generate the final prediction submission file in the [next section](#Late fusion of RGBD).

## Late fusion of RGBD
Prediction scores from RGB and depth modality are fused with weight 0.5.

1. For validation split (phase1):
```
python main_rgbd_fusion.py --score1 logs/test/score/focus_rgb_init_depth_val_score.txt \
 --score2 logs/test/score/focus_depth_init_rgb_val_score.txt --eval_split val
```
2. For test split (phase2):
```
python main_rgbd_fusion.py --score1 logs/test/score/focus_rgb_init_depth_test_score.txt \
 --score2 logs/test/score/focus_depth_init_rgb_test_score.txt --eval_split test
```
The prediction file in the required [format](https://competitions.codalab.org/competitions/16491#learn_the_details) will be located in `./logs/test/pred`.

## Training
A two-phase training procedure is adopted for RGB and Depth model, each phase will take 6-8 hours time on 4xTitan X(Maxwell) for each modality.

To train the model, follow these steps bellow:

1. Train RGB model initialized from C3D:
```
python main_train_rgbd_c3d.py --gpu_id 0,1,2,3 --modality focus_rgb
```
2. Train Depth model initialized from C3D:
```
python main_train_rgbd_c3d.py --gpu_id 0,1,2,3 --modality focus_depth
```
3. Finetune RGB model initialized with depth C3D:
```
python main_train_rgbd_c3d.py --gpu_id 0,1,2,3 --modality focus_rgb --resume ./model/focus_depth.model
```

4. Finetune depth model initialized with RGB C3D:
```
python main_train_rgbd_c3d.py --gpu_id 0,1,2,3 --modality focus_depth --resume ./model/focus_rgb.model
```

After the training, the final models should be found at:
```
model/focus_rgb_init_depth.model
model/focus_depth_init_rgb.model
```
 
## Performance
* Training time on 4 x Titan X(Maxwell) with batch size 32:  1.868s
* Testing time on 4 x Titan X(Maxwell) with batch size 32: 1.093s
* Accuracy on validation with RGBD late fusion(0.5RGB+0.5D): 0.6215
* Accuracy on test with RGBD late fusion(0.5RGB+0.5D): [TBD]
