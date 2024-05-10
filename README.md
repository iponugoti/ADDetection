# ADDetection
Alzheimer's Disease Detection - Enhanced Multimodal Approach: [Final Project for CS1470 Deep Learning](https://devpost.com/software/alzheimer-s-disease-detection-enhanced-multimodal-approach).

## Dataset
Our paper and model provides results on the [Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset](https://adni.loni.usc.edu/). The data is not provided in this repository and needs to be requested directly from ADNI.   

## Description
In this work, we presented a multi-modal, multi-class, attention-based deep learning framework to detect Alzheimer's disease using clinical and imaging (MRI) data from ADNI.

This repository contains the code both in TensorFlow (adapted from the MADDi model) and a PyTorch model (our final model that we implemented from scratch). Our final multimodal model is located in pytorch_training/train_all_modalities.py. 

## Preprocessing
To preprocess data, run the jupyter notebooks on ADNI data in the following order:
1. general (diagnosis making)
2. preprocess_clinical
3. preprocess_images
4. preprocess_training

## Training and Evaluation
To train and evaluate a uni-modal model baseline, run train_clinical.py or train_imaging.py.
To train and evaluate the multimodal architecture, run train_all_modalities.py.

## Credits
The original paper as well as some of the structure in this repo was adopted from https://github.com/rsinghlab/MADDi?tab=readme-ov-file

## Authors
[Isha Ponugoti](https://github.com/iponugoti)
[Karis Ma](https://github.com/karismajn)
[Raima Islam](https://github.com/raimaaislam)
[Timothy Pyon](https://github.com/timothypyon)
