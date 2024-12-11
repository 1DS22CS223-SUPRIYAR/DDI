# Drugram
:boom: The official repository of our paper "A Lightweight Self-Attention-Based Method Focused on Spatial Structure of Molecular Graph for Drug-Drug Interactions Prediction".

## Introduction
Multi-drug combinations for the treatment of complex diseases are gradually becoming an important treatment, and this type of treatment can take advantage of the synergistic effects among drugs. However, drug-drug interactions (DDIs) are not just all beneficial. Accurate and rapid identifications of the DDIs are essential to enhance the effectiveness of combination therapy and avoid unintended side effects. Traditional DDIs prediction methods use only drug sequence information or drug graph information, which ignores information about the position of atoms and edges in the spatial structure. In this paper, we propose Drugram, a method based on a lightweight attention mechanism for DDIs prediction. Drugram takes the two-dimension (2D) structures of drugs as input and encodes the molecular graph with spatial information. Besides, Drugram uses lightweight-based attention mechanism and self-attention distilling to process spatially the encoded molecular graph, which not only retains the multi-headed attention mechanism but also reduces the computational and storage costs. Finally, we use the siamese network architecture to serve as the architecture of Drugram, which can make full use of the limited data to train the model for better performance and also limit the differences to some extent between networks dealing with drug features.

## Overview

<p align="center">
  <img width="700" src="assets/overview.png" /> 
</p>

## Setup

A conda environment can be created with

`conda create --name Drugram python=3.7`

`conda activate Drugram`

`conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch`

## Training

`python train.py`

## Predicting

`python predict.py`

