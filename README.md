![neodroid](RepoAssets/images/header.png)

# Agent
This repository will host all initial machine learning efforts applying the [Neodroid](https://github.com/sintefneodroid/) platform.

<p align="center" width="100%">
  <a href="https://www.python.org/">
    <img alt="python" src="RepoAssets/images/python.svg" height="40" align="left">
  </a>
  <a href="https://opencv.org/" style="float:center;">
    <img alt="opencv" src="RepoAssets/images/opencv.svg" height="40" align="center">
  </a>
  <a href="http://pytorch.org/"style="float: right;">
    <img alt="pytorch" src="RepoAssets/images/pytorch.svg" height="40" align="right" >
  </a>
</p>
<p align="center" width="100%">
  <a href="http://www.numpy.org/">
    <img alt="numpy" src="RepoAssets/images/numpy.svg" height="40" align="left">
  </a>
  <a href="https://github.com/tqdm/tqdm" style="float:center;">
    <img alt="tqdm" src="RepoAssets/images/tqdm.gif" height="40" align="center">
  </a>
  <a href="https://matplotlib.org/" style="float: right;">
    <img alt="matplotlib" src="RepoAssets/images/matplotlib.svg" height="40" align="right">
  </a>
</p>

# Contents Of This Readme
- [Models](#models)
  - [Target Point Estimator](#target-point-estimator)
  - [Perfect Information Navigator](#perfect-information-navigator)
- [Requirements](#requirements)
- [Usage](#usage)
- [To Do](#to-dos)
- [Contributing](#contributing)
- [Other Components](#other-components-of-the-neodroid-platform)

# Models

## Target Point Estimator
Using Depth, Segmentation And RGB images to estimate the location of target point in an environment.

## Perfect Information Navigator
Has access to perfect location information about the obstructions and target in the environment, the objective is to navigate to the target with colliding with the obstructions.

# Requirements
- pytorch
- neodroid

(Optional)
- visdom

To install these use the command:
````bash
pip3 install -r requirements.txt
````

# Usage
For training a model use:
````bash
python3 train_model.py
````
For testing a trained model use:
````bash
python3 test_model.py
````

# To Do's
- [ ] Actually make the models work

# Contributing
See guidelines for contributing [here](CONTRIBUTING.md).


# Other Components Of the Neodroid Platform

- [neo](https://github.com/sintefneodroid/neo)
- [droid](https://github.com/sintefneodroid/droid)
