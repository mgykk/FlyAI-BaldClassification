# -*- coding: utf-8 -*
import sys
import os

DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
DataID="BaldClassification"
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')
MODELS = ['inceptionv4','efficientnet-b0']
