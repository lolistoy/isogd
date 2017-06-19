## Dependencies
Install python depencies: cv2, h5py, numpy, PIL, pytorch 
```
sudo apt-get install python-opencv
sudo pip install h5py, numpy, Pillow
sudo pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
sudo pip install torchvision
```

## Model and Data
1. Replace soft-links `IsoGD_phase_1` and `IsoGD_phase_2` in `dataset` 
2. Download pre-trained model `model_data.tar.gz` from 
[rgbd model and preprocessed pose data](https://www.dropbox.com/s/zn1guimo7puznb4/model_data.tar.gz?dl=0).
3. Uncompress the model `tar xf model_data.tar.gz`

## Evaluation
Prediction results used for submission can be found in [pred](logs/test/pred).
 
1. To evaluate on validation split(phase1) with RGB modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_rgb_init_depth.model \
--gpu_id 0,1 --evaluate 1 --eval_split val --modality focus_rgb 
```
2. To evaluate on validation split(phse1) with depth modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_rgb_init_depth.model \
--gpu_id 0,1 --evaluate 1 --eval_split val --modality focus_depth 
```
3. To evaluate on test split(phase2) with RGB modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_rgb_init_depth.model \
--gpu_id 0,1 --evaluate 1 --eval_split test --modality focus_rgb 
```
4. To evaluate on test split(phse2) with depth modality:
```
python main_train_rgbd_c3d.py --resume ./model/focus_rgb_init_depth.model \
--gpu_id 0,1 --evaluate 1 --eval_split test --modality focus_depth 
```

## Late fusion of RGBD
1. For validation split(phase1):
```
python main_rgbd_fusion.py --score1 logs/test/score/focus_rgb_val_score.txt \
 --score2 logs/test/score/focus_depth_val_score.txt --eval_split val
```
2. For test split(phase2):
```
python main_rgbd_fusion.py --score1 logs/test/score/focus_rgb_test_score.txt \
 --score2 logs/test/score/focus_depth_test_score.txt --eval_split test
```
The prediction file in the required [format](https://competitions.codalab.org/competitions/16491#learn_the_details) will be located in `./logs/test/pred`.

## Performance
* Training time on 4 x Titan X(Maxwell) with batch size 32:  1.868s
* Testing time on 4 x Titan X(Maxwell) with batch size 32: 1.093s
* Accuracy on validation with RGBD late fusion(0.5RGB+0.5D): 0.6184
* Accuracy on test with RGBD late fusion(0.5RGB+0.5D): [TBD]

