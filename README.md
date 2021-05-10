# Primitive Representation Learning Network (PREN)
This repository contains the code for our paper 

**"Primitive Representation Learning for Scene Text Recognition" (accepted by CVPR 2021)**

For now only code for PREN is provided, code for PREN2D is being sorted out.

## Requirements

- python 3.7.9, pytorch 1.4.0, and torchvision 0.5.0 (other versions will probably work but not tested)
- other libaries can be installed by
```
pip install -r requirements.txt
```

## Recognition with pretrained model

We provide code for using our pretrained model to recognize text images.

To recognize three samples in the "samples" folder, just run 
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

1. Modify training configurations in ```Configs/trainConf.py```
2. Run ```python train.py```

To run the training code, one can only modify ```image_dir``` and ```train_list``` to his own training data. 

```image_dir``` is the path of training data root (e.g., ./samples)

```train_list``` is the path of a text file containing image paths (relative to ```image_dir```) and corresponding labels.

E.g., ```train_list``` can be a text file with the following content:

```
001.jpg RONALDO
002.png LEAVES
003.jpg SALMON
```

## Test
Similar to train, one can modify ```Configs/testConf.py``` and run ```python test.py``` to evaluate a model.

## Citation
If you find this project helpful for your research, please cite our paper:

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
