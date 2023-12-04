
# CyberTruck - Distracted Driver Detection

## Motivation

insert image here ----

Developing an AI tool for identifying distracted driving is motivated by the urgent need to enhance road safety. With the escalating incidents of distractions, particularly due to technological devices, leveraging advanced AI, computer vision, and machine learning technologies becomes essential. The goal is to proactively prevent accidents by real-time detection of distracted behaviors, providing immediate alerts to drivers or triggering safety systems. This initiative aligns with evolving regulations, contributes to public awareness, and fosters collaboration for a comprehensive approach to road safety. Ultimately, the AI tool aims to save lives, reduce accidents, and promote responsible driving habits.

## Our work

insert images -----

This model is built upon the foundation laid out in the NeuroIPS 2018 paper, specifically titled "Driver Distraction Identification with an Ensemble of Convolutional Neural Networks." Our team has undertaken the task of replicating this model, introducing certain modifications, and incorporating our unique version of a genetic algorithm.

One notable alteration made by our team involves the adjustment of the number of classifiers, a component originally presented in the referenced paper. This modification was implemented to tailor the model to our specific requirements and potentially enhance its performance.

It is essential to note that the successful implementation of our model relies on obtaining the dataset provided by https://heshameraqi.github.io/distraction_detection. This dataset serves as a crucial resource, enabling us to train and evaluate the model effectively. The utilization of this dataset ensures the model's exposure to diverse real-world scenarios, contributing to its robustness and efficacy in identifying driver distractions.

# Repo setup

- `detection` - Contains all the python files for the detection model
  - `face_detection` code pertaining to detecting facial features.
  - `hands_detection` code pertaining to detecting hand features.
  - `model.py`
- `client` - Contains all of the code for the android app.

If you want to add a new module create a folder with an `__init__.py` file and import code into it that you want to expose.

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

To install Cybertruck and its depenencies, we strongly recommend the use of a package manager like conda:

```
git clone cap6411-cybertruck/cybertruck
conda create -n cyber python=3.10
pip install -r requirements.txt
```


## DEMO


CAP6411 Fall 2023 - Group 6   
Ron, Robin, Suneet, Osi, Kasun




##Dataset
