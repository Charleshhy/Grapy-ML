# Grapy-ML: Graph Pyramid Mutual Learning for Cross-dataset Human Parsing

This repository contains pytorch source code for AAAI2020 oral paper: [Grapy-ML: Graph Pyramid Mutual Learning for Cross-dataset Human Parsing](https://arxiv.org/abs/1911.12053) by Haoyu He, Jing Zhang, Qiming Zhang and Dacheng Tao.

* * *
## Grapy-ML:

![GPM](images/2317-Figure2.jpg?raw=true "Title")

* * *

## Getting Started:

### Environment:
 - Pytorch = 1.1.0

 - torchvision

 - scipy

 - tensorboardX

 - numpy

 - opencv-python

 - matplotlib

### Data Preparation:

You need to download the three datasets. The CIHP dataset and ATR dataset can be found in [this](https://github.com/Gaoyiminggithub/Graphonomy) repository and our code is heavily borrowed from it as well.

Then, the datasets should be arranged in the following folder, and images should be rearranged with the provided file structure.

    /data/dataset/
    
### Testing:


The pretrain models and some trained models are provided [here](https://drive.google.com/drive/folders/1eQ9IV4QmcM5dLCuVMSVE3ogVpL6qUQL5?usp=sharing) for testing and training.

| Model Name        | Description           | Derived from |
| :-------------: |:-------------:|:-------------:|
| deeplab_v3plus_v3.pth      | The Deeplab v3+'s pretrain weights | |
| CIHP_pretrain.pth      |  The reproduced Deeplab v3+ model trained on CIHP dataset  | deeplab_v3plus_v3.pth |
| CIHP_trained.pth | GPM model trained on CIHP dataset    | CIHP_pretrain.pth |
| deeplab_multi-dataset.pth | The reproduced multi-task learning Deeplab v3+ model trained on CIHP, PASCAL-Person-Part and ATR dataset | deeplab_v3plus_v3.pth |
| GPM-ML_multi-dataset.pth | Grapy-ML model trained on CIHP, PASCAL-Person-Part and ATR dataset | deeplab_multi-dataset.pth |
| GPM-ML_finetune_PASCAL.pth | Grapy-ML model finetuned on PASCAL-Person-Part dataset | GPM-ML_multi-dataset.pth |


To test, run the following two scripts:
    
    bash eval_gpm.sh
    bash eval_gpm_ml.sh


### Training:

#### GPM:
During training, you first need to get the Deeplab pretrain model(e.g. CIHP_dlab.pth) on each dataset. Such act aims to provide a trustworthy initial raw result for the GSA operation in GPM.
   
    bash train_dlab.sh
The imageNet pretrain model is provided in the following table, and you should swith the dataset name and target classes to the dataset you want in the script. (CIHP: 20 classes, PASCAL: 7 classes and ATR: 18 classes)

In the next step, you should utilize the Deeplab pretrain model to further train the GPM model.

    bash train_gpm.sh 

It is recommended to follow the training settings in our paper to reproduce the results.



#### GPM-ML: 
  
Firstly, you can conduct the deeplab pretrain process by the following script:
   
    bash train_dlab_ml.sh
The multi-dataset Deeplab V3+ is transformed as a simple multi-task task.

Then, you can train the GPM-ML model with the training set from all three datasets by:

    bash train_gpm_ml_all.sh
After this phase, the first two levels of the GPM-ML model would be more robust and generalized.

Finally, you can try to finetune on each dataset by the unified pretrain model.

    bash train_gpm_ml_pascal.sh
    
### Citation:

    @inproceedings{he2020grapy,
    title={Grapy-ML: Graph Pyramid Mutual Learning for Cross-dataset Human Parsing},
    author={He, Haoyu and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2020}
    }
    
### Maintainer:
haoyu.he@sydney.edu.au
