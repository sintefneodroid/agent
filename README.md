![neodroid](.github/images/header.png)

# Agent
This repository will host all initial machine learning efforts applying the [Neodroid](https://github.com/sintefneodroid/) platform.


<table>
   <tr>
     <td>
       <a href='https://travis-ci.org/sintefneodroid/agent'>
         <img src='https://travis-ci.org/sintefneodroid/agent.svg?branch=master' alt='Build Status' />
       </a>
     </td>
     <td>
       <a href='https://coveralls.io/github/sintefneodroid/agent?branch=master'>
         <img src='https://coveralls.io/repos/github/sintefneodroid/agent/badge.svg?branch=master' alt='Coverage Status' />
       </a>
     </td>
     <td>
       <a href='https://github.com/sintefneodroid/agent/issues'>
         <img src='https://img.shields.io/github/issues/sintefneodroid/agent.svg?style=flat' alt='GitHub Issues' />
       </a>
     </td>
     <td>
       <a href='https://github.com/sintefneodroid/agent/network'>
         <img src='https://img.shields.io/github/forks/sintefneodroid/agent.svg?style=flat' alt='GitHub Forks' />
       </a>
     </td>
       <td>
       <a href='https://github.com/sintefneodroid/agent/stargazers'>
         <img src='https://img.shields.io/github/stars/sintefneodroid/agent.svg?style=flat' alt='GitHub Stars' />
       </a>
     </td>
       <td>
       <a href='https://github.com/sintefneodroid/agent/blob/master/LICENSE.md'>
         <img src='https://img.shields.io/github/license/sintefneodroid/agent.svg?style=flat' alt='GitHub License' />
       </a>
     </td>
  </tr>
</table>

<p align="center" width="100%">
  <a href="https://www.python.org/">
    <img alt="python" src=".github/images/python.svg" height="40" align="left">
  </a>
  <a href="https://opencv.org/" style="float:center;">
    <img alt="opencv" src=".github/images/opencv.svg" height="40" align="center">
  </a>
  <a href="http://pytorch.org/"style="float: right;">
    <img alt="pytorch" src=".github/images/pytorch.svg" height="40" align="right" >
  </a>
</p>
<p align="center" width="100%">
  <a href="http://www.numpy.org/">
    <img alt="numpy" src=".github/images/numpy.svg" height="40" align="left">
  </a>
  <a href="https://github.com/tqdm/tqdm" style="float:center;">
    <img alt="tqdm" src=".github/images/tqdm.gif" height="40" align="center">
  </a>
  <a href="https://matplotlib.org/" style="float: right;">
    <img alt="matplotlib" src=".github/images/matplotlib.svg" height="40" align="right">
  </a>
</p>

# Contents Of This Readme
- [Models](#models)
  - [Target Point Estimator](#target-point-estimator)
  - [Perfect Information Navigator](#perfect-information-navigator)
- [Requirements](#requirements)
- [Usage](#usage)
- [To Do's](#to-do's)
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
Export python path to the repo root so we can use the utilities module
````bash
export PYTHONPATH=/path-to-repo/
````
For training a agent use:
````bash
python3 procedures/train_agent.py
````
For testing a trained agent use:
````bash
python3 procedures/test_agent.py
````

# To Do's
- [ ] Actually make the models work

# Contributing
See guidelines for contributing [here](CONTRIBUTING.md).


# Other Components Of the Neodroid Platform

- [neo](https://github.com/sintefneodroid/neo)
- [droid](https://github.com/sintefneodroid/droid)
