#!/bin/bash

conda create -y -n rsna2 python=3.6
conda activate rsna2

conda install -y -n rsna2 pytorch=0.4.1 cuda90 -c pytorch
pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -y -c conda-forge pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple
