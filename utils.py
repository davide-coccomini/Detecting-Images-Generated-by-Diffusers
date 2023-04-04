import cv2
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
import numpy as np
import os
import cv2
import torch
from statistics import mean

approaches = {1: "FaceShifter", 2: "FS-GAN", 3: "DeepFakes", 4: "BlendFace", 5: "MMReplacement", 6: "DeepFakes-StarGAN-Stack", 7: "Talking Head Video", 8: "ATVG-Net", 9: "StarGAN-BlendFace-Stack", 10: "First Order Motion", 11: "StyleGAN2", 12: "MaskGAN", 13: "StarGAN2", 14: "SC-FEGAN", 15: "DiscoFaceGAN"}
groups = {"Transfer": [1, 2, 3], "Swap": [4, 5], "FSM": [6, 9], "Face_Reenactment": [7, 8, 10], "Face Editing": [11, 12, 13, 14, 15]}


def transform_frame(image, image_size):
    transform_pipeline = Compose([
                IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REPLICATE)
                ]
            )
    return transform_pipeline(image=image)['image']
    
    
def resize(image, image_size):
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except:
        return []

def custom_round(values):
    result = []
    for value in values:
        if value > 0.55:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class

def custom_round(pred):
    if pred > 0.5:
        return 1.
    else:
        return 0.

def multiple_custom_round(values):
    result = []
    for value in values:
        if value > 0.6:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)


def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img