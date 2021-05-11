# Primitive Representation Learning Network (PREN)
This repository contains the code for our paper 

> [Primitive Representation Learning for Scene Text Recognition](https://arxiv.org/pdf/2105.04286.pdf)

> Ruijie Yan, Liangrui Peng, Shanyu Xiao, Gang Yao

For now only code for PREN is provided, code for PREN2D is being sorted out.

## Requirements

- python 3.7.9, pytorch 1.4.0, and torchvision 0.5.0 (other versions may probably work, but didn't being tested)
- other libaries can be installed by
```
pip install -r requirements.txt
```

## Recognition with pretrained model

We provide code for using our pretrained model to recognize text images.

- The pretrained model can be downloaded via Baidu net disk: [download_link](https://pan.baidu.com/s/1iHc_F2pNUS1_QwBUaMrxvw) key: 2txt

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

## Train
Two simple steps to train your own model:

- Modify training configurations in ```Configs/trainConf.py```
- Run ```python train.py```

To run the training code, one can only modify ```image_dir``` and ```train_list``` to his own training data. 

```image_dir``` is the path of training data root.

```train_list``` is the path of a text file containing image paths (relative to ```image_dir```) and corresponding labels.

For example, ```image_dir``` could be ```'./samples'```, and ```train_list``` could be a text file with the following content

```
001.jpg RONALDO
002.png LEAVES
003.jpg SALMON
```

## Test
Similar to train, one can modify ```Configs/testConf.py``` and run ```python test.py``` to evaluate a model.

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
  year      = {2021}
}
```
