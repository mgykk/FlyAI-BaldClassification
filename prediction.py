# -*- coding: utf-8 -*
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch.functional import img_to_tensor
from flyai.framework import FlyAI
from torch.autograd import Variable
from models.modelzoo.senet2 import seresnext26_32x4d, seresnet34
from pytorch_toolbelt.inference import tta

from path import MODEL_PATH, DATA_PATH, MODELS
import cv2 as cv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Prediction(FlyAI):
    def __init__(self):
        self.model = None
        
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.models = []
        for m in MODELS:
            for fold in range(5):
                submodel = torch.load(MODEL_PATH + '/' + f"{m}_best_fold{fold+1}.pth")
                self.models.append(submodel.to(device))
            
    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/data\/input\/BaldClassification\/image\/0.png"}
        :return: 模型预测成功之后返回给系统样例 {"label":"0"}
        '''
        print(image_path)
        img = cv.imread(image_path)
        img = cv.resize(img, dsize=(160, 160))
        tensor = img_to_tensor(img)
        tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)

        result = []
        for subModel in self.models:
            output = tta.fliplr_image2label(subModel, tensor.to(device))
            result.append(output)
        new_output = torch.mean(torch.stack(result, 0), 0)
        pred = new_output.max(1, keepdim=True)[1]

        # output = tta.fliplr_image2label(self.model, tensor.to(device))
        # pred = output.max(1, keepdim=True)[1].item()

        return {"label": pred}

