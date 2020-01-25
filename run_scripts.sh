#!/bin/sh
python train_models.py --pyaml '/experiments/resnet50_config.yaml'
python train_models.py --pyaml '/experiments/densenet121_config.yaml'
#python train_models.py --pyaml '/experiments/efficientnetb2_config.yaml'
#python train_models.py --pyaml '/experiments/efficientnetb3_config.yaml'
python train_models.py --pyaml '/experiments/efficientnetb4_config.yaml'
