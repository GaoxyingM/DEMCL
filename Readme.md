# DEMCL
title: A Diffusion-Enhanced Mixed Contrastive Learning framework for Bundle Recommendation

This is our Pytorch implementation for the paper:

## Requirements
* python == 3.8.12 
* supported(tested) CUDA versions: 10.1
* Pytorch == 1.4.0 or above

## Code Structure
1. The entry script for training and evaluation is: [train.py]
2. The config file is: [config.yaml]
3. The script for data preprocess and dataloader: [utility.py]
4. The model folder: [./models]
5. The experimental logs in tensorboard-format are saved in [./runs.]
6. The experimental logs in txt-format are saved in [./log.]
7. The best model and associate config file for each experimental setting is saved in [./checkpoints.]

# Hor to run the code
1. Decompress the dataset file into the current folder:
   > tar -zxcf dataset.tgz
   
   Note: For the iFashion dataset, we include three additional files: user\_id\_map.json, item\_id\_map.json, and bundle\_id\_map.json. These files provide mappings between the original string-formatted IDs in the POG dataset and the integer-formatted IDs used in our dataset. These mappings can be used to retrieve the original content information of the items or outfits. However, no content information is utilized in our study.
   

2. Train DEMCL on the dataset Youshu with GPU 1:
   > python train.py -g 1 -d Youshu -m DEMCL
   
    
	You can specify the GPU ID and dataset via command-line arguments, while hyper-parameters can be adjusted by modifying the configuration file [config.yaml]. A detailed description of the hyper-parameters is provided in the config file. For a deeper understanding of the impact of key hyper-parameters, we strongly recommend reading the associated paper.
   
