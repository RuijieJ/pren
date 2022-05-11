# Primitive Representation Learning Network (PREN)
This repository contains the code for our paper accepted by CVPR 2021

> [Primitive Representation Learning for Scene Text Recognition](https://arxiv.org/abs/2105.04286)

> Ruijie Yan, Liangrui Peng, Shanyu Xiao, Gang Yao

For now we only provide code for PREN.

## Requirements

- python 3.7.9, pytorch 1.4.0, and torchvision 0.5.0
- other libraries can be installed by
```
pip install -r requirements.txt
```

## Recognition with pretrained model

We provide code for using our pretrained model to recognize text images.

- The pretrained model can be downloaded via [Google drive](https://drive.google.com/file/d/1lwDlD3gLqeX4t9EEIib-JUlVWOeX8JPL/view?usp=sharing) or [Baidu net disk](https://pan.baidu.com/s/1iHc_F2pNUS1_QwBUaMrxvw) (key: 2txt)

- After downloading the pretrained model (pren.pth), put it in the **"models"** folder.

- To recognize three samples in the "samples" folder, just run 
```python
python recog.py
```

The results would be
```
[Info] Load model from ./models/pren.pth
samples/001.jpg: ronaldo
samples/002.png: leaves
samples/003.jpg: salmon
```

## Training
Two simple steps to train your own model:

- Modify training configurations in ```Configs/trainConf.py```
- Run ```python train.py```

To run the training code, please modify ```image_dir``` and ```train_list``` to your own training data. 

```image_dir``` is the path of training data root.

```train_list``` is the path of a text file containing image paths (relative to ```image_dir```) and corresponding labels.

For example, ```image_dir``` could be ```'./samples'```, and ```train_list``` could be a text file with the following content

```
001.jpg RONALDO
002.png LEAVES
003.jpg SALMON
```

## Evaluation
Similar to train, one can modify ```Configs/testConf.py``` and run ```python test.py``` to evaluate a model.

## Acknowledgement
The code of EfficientNet is modified from [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch), where we output multi-scale feature maps.

## Citation
If you find this project helpful for your research, please cite our paper

```
@inproceedings{yan2021primitive,
  author    = {Yan, Ruijie and
               Peng, Liangrui and
               Xiao, Shanyu and
               Yao, Gang},
  title     = {Primitive Representation Learning for Scene Text Recognition},
  booktitle = {CVPR},
  year      = {2021},
  pages     = {284-293}
}
```
