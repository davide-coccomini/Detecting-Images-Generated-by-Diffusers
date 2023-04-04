import torch
import random
import yaml
import argparse
import pandas as pd
import os
from os import cpu_count
import cv2
import numpy as np
import math
from transforms.albu import IsotropicResize
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from pytorch_lightning import seed_everything
import clip
from sklearn.model_selection import train_test_split
import seaborn as sns
import csv
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from numpy import loadtxt
from utils import custom_round, multiple_custom_round
import spacy
import matplotlib.pyplot as plt
import en_core_web_sm
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from albumentations import Compose, PadIfNeeded

from sklearn.model_selection import train_test_split
import collections
from images_dataset import ImagesDataset
from progress.bar import ChargingBar
from utils import check_correct, resize, get_n_params, center_crop
#from transformers import AutoImageProcessor, SwinModel
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from collections import Counter

IMAGE_SIZE = 224

def read_images(caption, dataset, data_path, mode, transform):
    caption_path = os.path.join(data_path, caption)
    #caption = clip.tokenize([str(caption)])
    images_paths = os.listdir(caption_path)
    for image_name in images_paths:
        if "tags.txt" in image_name:
            continue
        if "real" in image_name:
            label = 0
        else:
            try:
                if "-" in image_name:
                    generation = int(image_name.split("-")[1].split(".")[0])
                else:
                    generation = int(image_name.split(".")[0])
            except:
                print(caption, image_name)
            if generation == 0:
                label = 1
            else:
                continue

        image_path = os.path.join(caption_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        if opt.analyze_tags:
            tags_path = os.path.join(caption_path.replace("glide_", ""), "tags.txt")
            tags = []
            with open(tags_path, 'r') as fd:
                reader = csv.reader(fd)
                for row in reader:
                    tags.append(row[0])
        
        image = center_crop(image)
        image = transform(image=image)['image']
        
        if opt.analyze_tags:
            row = (image, caption, label, tags)
        else:
            row = (image, caption, label)
        dataset.append(row)

def get_text_features(caption, nlp):
    # Features Vector
    features = {"LENGTH": len(caption),
                "ADJ": 0,
                "ADP": 0,
                "ADV": 0,
                "AUX": 0,
                "CCONJ": 0,
                "DET": 0,
                "INTJ": 0,
                "NOUN": 0,
                "NUM": 0,
                "PART": 0,
                "PRON": 0,
                "PROPN": 0,
                "PUNCT": 0,
                "SCONJ": 0,
                "SYM": 0,
                "VERB": 0,
                "X": 0,
                "SPACE": 0,
                "STOPS": 0,
                "NON_ALPHA": 0,
                "NAMED_ENTITIES": 0,
                "LABEL": 0,
                "BINARY_LABEL": 0}

    caption = nlp(caption)
    for token in caption:
        pos = token.pos_
        features[pos] += 1
        if token.is_stop:
            features["STOPS"] += 1
        if not token.is_alpha:
            features["NON_ALPHA"] += 1

    for ent in caption.ents: 
        if ent.label_ != "":
            features["NAMED_ENTITIES"] += 1

    return list(features.values()), list(features.keys())

def create_pre_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

# Main body
if __name__ == "__main__":
    seed_everything(42)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=100, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--test_path', default='../datasets/diffused_coco/test', type=str,
                        help='Test directory')
    parser.add_argument('--model_weights', default='', type=str, metavar='PATH',
                        help='Path to the checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--display_pre', type=str, default='',
                        help='Display pre.')
    parser.add_argument('--display_post', type=str, default='',
                        help='Display post.')
    parser.add_argument('--gpu_id', default=6, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--config', type=str, default='',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: SwinViT, 1: Resnet50, 2: CLIP+MLP; 3: CLIP+LMLP)")
    parser.add_argument('--clip_mode', type=int, default=0, 
                        help="Which model architecture version to be trained (0: Resnet50, 1: ViT)")
    parser.add_argument('--mode', type=int, default=0, 
                        help="Which mode to be used (0: Image-Only, 1: Image+Text[Only CLIP])")
    parser.add_argument('--show_stats',  action="store_true", default=False, 
                        help="Show stats")
    parser.add_argument('--analyze_tags', action="store_true", default=False, 
                        help="Use tags.txt for error analysis.")
                        
    opt = parser.parse_args()
    print(opt)

    # Model Loading
    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
    if opt.mode == 0 and opt.model < 2:
        if opt.model == 0: 
            HUB_URL = "SharanSMenon/swin-transformer-hub:main"
            MODEL_NAME = "swin_tiny_patch4_window7_224"
            model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)
            model.head = torch.nn.Linear(768, config['model']['num-classes'])
        elif opt.model == 1: 
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
    else:
        if opt.clip_mode == 0:
            clip_model, preprocess = clip.load("RN50", device='cpu')
            dim = 1
        else:
            clip_model, preprocess = clip.load("ViT-B/32", device=torch.device('cpu'))
            dim = 0.5
        clip_model = clip_model.float()
        clip_model.to(opt.gpu_id)
        clip_model.eval()
        if opt.model == 2:
            if opt.mode == 0:
                model = torch.nn.Linear(int(1024*dim), config['model']['num-classes'])
            else:
                model = torch.nn.Linear(int(2048*dim), config['model']['num-classes'])
        elif opt.model == 3:
            if opt.mode == 0:
                model = torch.nn.Sequential(torch.nn.Linear(int(1024*dim), 512),
                                            torch.nn.Linear(512, 512),
                                            torch.nn.Linear(512, config['model']['num-classes']))
            else:
                model = torch.nn.Sequential(torch.nn.Linear(int(2048*dim), 512),
                                            torch.nn.Linear(512, 512),
                                            torch.nn.Linear(512, config['model']['num-classes']))
        elif opt.model == 4:
            if opt.mode == 0:
                model = torch.nn.Sequential(torch.nn.Linear(int(1024*dim), 512),
                                            torch.nn.Linear(512, 512),
                                            torch.nn.Linear(512, 256),
                                            torch.nn.Linear(256, 256),
                                            torch.nn.Linear(256, config['model']['num-classes']))
            else:
                model = torch.nn.Sequential(torch.nn.Linear(int(2048*dim), 512),
                                            torch.nn.Linear(512, 512),
                                            torch.nn.Linear(512, 256),
                                            torch.nn.Linear(256, 256),
                                            torch.nn.Linear(256, config['model']['num-classes']))
        elif opt.model == 5:
            if opt.mode == 0:
                model = torch.nn.Sequential(torch.nn.Linear(int(1024*dim), 4096),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.Linear(4096, 1024),
                                            torch.nn.Linear(1024, config['model']['num-classes']))
            else:
                model = torch.nn.Sequential(torch.nn.Linear(int(2048*dim), 4096),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.Linear(4096, 1024),
                                            torch.nn.Linear(1024, config['model']['num-classes']))
        elif opt.model == 6:
            if opt.mode == 0:
                model = torch.nn.Sequential(torch.nn.Linear(int(1024*dim), 8192),
                                            torch.nn.Linear(8192, 4096),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.Linear(4096, 2048),
                                            torch.nn.Linear(2048, 2048),
                                            torch.nn.Linear(2048, 1024),
                                            torch.nn.Linear(1024, 1024),
                                            torch.nn.Linear(1024, config['model']['num-classes']))
            else:
                model = torch.nn.Sequential(torch.nn.Linear(int(2048*dim), 8192),
                                            torch.nn.Linear(8192, 4096),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.Linear(4096, 2048),
                                            torch.nn.Linear(2048, 2048),
                                            torch.nn.Linear(2048, 1024),
                                            torch.nn.Linear(1024, 1024),
                                            torch.nn.Linear(1024, config['model']['num-classes']))
        elif opt.model == 7:
            model = timm.create_model('xception', pretrained=True)
            model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
    print("*******************************************************************************************************************")
    print(os.path.exists(opt.model_weights))
    if opt.model_weights != '':
        model_weights = opt.model_weights
        while not os.path.exists(model_weights):
            epoch = int(model_weights.split("_")[-1])
            model_name = '_'.join(model_weights.split("_")[:-1])
            model_weights = model_name + "_" + str(epoch - 1)
            print("Trying new model weights", model_weights)
            if epoch == 0:
                print("No model found.")
                exit()
        model.load_state_dict(torch.load(model_weights, map_location='cpu'))
        print("Weights loaded")
    else:
        print("No weights loaded.")
        exit()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters:", params)

    test_captions = os.listdir(opt.test_path)
    mgr = Manager()
    test_dataset = mgr.list()
    transform = create_pre_transform(IMAGE_SIZE)

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(test_captions)) as pbar:
            for v in p.imap_unordered(partial(read_images, dataset=test_dataset, data_path = opt.test_path, mode = opt.mode, transform=transform), test_captions):
                pbar.update()

    if opt.analyze_tags:
        test_tags = [row[3] for row in test_dataset]
    test_labels = [float(row[2]) for row in test_dataset]
    test_captions = [row[1] for row in test_dataset]
    test_dataset = [row[0] for row in test_dataset]

    test_samples = len(test_dataset)

    # Print some useful statistics
    print("Test images:", len(test_dataset))
    print("__TEST STATS__")
    test_counters = collections.Counter(test_labels)
    print(test_counters)

    # Create the data loaders
    if opt.config != '':
        batch_size = config['test']['bs']
    else:
        batch_size = 8
        
    test_dataset = ImagesDataset(np.asarray(test_dataset), test_captions, np.asarray(test_labels), IMAGE_SIZE, mode = 'val')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del test_dataset

    model = model.to(opt.gpu_id)
    model.eval()   
    preds = []
    bar = ChargingBar('PREDICT', max=(len(test_dl)))

    for index, (images, captions, labels) in enumerate(test_dl):
        images = np.transpose(images, (0, 3, 1, 2))
        labels = labels.unsqueeze(1)
        images = images.to(opt.gpu_id)
        captions = captions.to(opt.gpu_id)
        with torch.no_grad():
            if opt.model < 2 or opt.model == 7:
                test_pred = model(images)
            else:
                image_features = clip_model.encode_image(images)
                if opt.mode == 1:
                    text_features = clip_model.encode_text(captions)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    features = torch.cat((image_features, text_features), dim = 1)
                else:
                    features = image_features
                features = features.float()
                features = torch.nn.functional.normalize(features)
                test_pred = model(features)
            test_pred = test_pred.cpu()
            preds.extend(test_pred)
            bar.next()
    bar.finish()
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()) for pred in preds]
    
    fpr, tpr, th = metrics.roc_curve(test_labels, preds)
    auc = metrics.auc(fpr, tpr)
    preds = multiple_custom_round(np.asarray(preds))
    accuracy = accuracy_score(preds, test_labels)

    print("METRICS")
    print("AUC", auc, "accuracy", accuracy)
    print(opt.display_pre, str(round(accuracy*100,2)), "|", str(round(auc*100,2)), opt.display_post)
    # ERROR ANALYSIS
    if opt.analyze_tags:
        categories = [item for sublist in test_tags for item in sublist]
        categories_counter = collections.Counter(categories)
        categories_errors = {key: [0, 0] for key in set(categories)}
    captions_errors = {key: [0, 0] for key in set(test_captions)}
    for i in range(len(preds)):
        pred = preds[i]
        if pred != test_labels[i]: # This image has been missclassified
            caption = test_captions[i]
            if opt.analyze_tags:
                tags = test_tags[i]
            if test_labels[i] == 0: # False positive
                captions_errors[caption][0] += 1
                if opt.analyze_tags:
                    for tag in tags:
                        categories_errors[tag][0] += 1
            else: # False negative
                captions_errors[caption][1] += 1
                if opt.analyze_tags:
                    for tag in tags:
                        categories_errors[tag][1] += 1
    
    if opt.analyze_tags:
        # Clean irrelevant categories
        categories = list(categories_errors.keys())
        for tag in categories:
            if categories_counter[tag] < 5:
                del categories_errors[tag]

    

        # Convert counters to percentage
        categories = list(categories_errors.keys())
        for tag in categories:
            categories_errors[tag][0] = int(categories_errors[tag][0] * 100 / categories_counter[tag])
            categories_errors[tag][1] = int(categories_errors[tag][1] * 100 / categories_counter[tag])

        # Get average macro category errors
        inanimate_categories = ["Location","Building", "Food", "Restaurant", "Island", "Event", "University", "NaturalPlace", "Automobile", "Infrastructure", "Organisation", "Town", "River", "Structure", "RouteOfTransportation", "Device", "Weapon", "BodyOfWater", "Stream", "Road", "HistoricPlace", "Village", "Software", "HistoricBuilding", "Drug", "Event", "Song", "ShoppingMall", "Hotel", "Castle", "ArtificialSatellite", "Motorcycle", "Bridge", "Aircraft", "ArchitecturalStructure", "Place", "RouteOfTransportation", "Plant", "MeanOfTransportation"]
        animate_categories = ["Politician", "OfficeHolder", "Cyclist", "SoccerPlayer", "MilitaryPerson", "Species", "Scientist", "Artist", "Athlete", "SoccerPlayer", "Eukaryote", "Agent", "Person", "Writer", "MusicalArtist", "Astronaut", "SportsManager", "Cleric", "Mammal", "AmericanFootballPlayer", "MilitaryUnit"]
        
        macro_categories_errors = {"animate": [[], 0], "inanimate": [[], 0]}
        for tag in categories_errors:
            if tag in inanimate_categories:
                macro_categories_errors["inanimate"][0].append(categories_errors[tag][1])
            if tag in animate_categories:
                macro_categories_errors["animate"][0].append(categories_errors[tag][1])
        macro_categories_errors["inanimate"][0] = sorted(macro_categories_errors["inanimate"][0], reverse=True)[:7]
        macro_categories_errors["animate"][0] = sorted(macro_categories_errors["animate"][0], reverse=True)[:7]
        
        for key in macro_categories_errors:
            macro_categories_errors[key][1] = sum(macro_categories_errors[key][0]) / len(macro_categories_errors[key][0])
        for key in categories_errors:
            print(key, categories_errors[key])

        print("Errors by macro category FALSE NEGATIVE")
        print(macro_categories_errors)

        macro_categories_errors = {"animate": [[], 0], "inanimate": [[], 0]}
        for tag in categories_errors:
            if tag in inanimate_categories:
                macro_categories_errors["inanimate"][0].append(categories_errors[tag][0])
            if tag in animate_categories:
                macro_categories_errors["animate"][0].append(categories_errors[tag][0])
        macro_categories_errors["inanimate"][0] = sorted(macro_categories_errors["inanimate"][0], reverse=True)[:7]
        macro_categories_errors["animate"][0] = sorted(macro_categories_errors["animate"][0], reverse=True)[:7]
        
        for key in macro_categories_errors:
            macro_categories_errors[key][1] = sum(macro_categories_errors[key][0]) / len(macro_categories_errors[key][0])
        for key in categories_errors:
            print(key, categories_errors[key])

            
        print("Errors by macro category FALSE POSITIVE")
        print(macro_categories_errors)
            
        # Clean irrelevant categories
        for tag in categories:
            if categories_errors[tag][0] + categories_errors[tag][1] < 2:
                del categories_errors[tag]
        
        categories_errors = {k: v for k, v in sorted(categories_errors.items(), key=lambda item: item[1][1], reverse=True)}


      
    # Plots
    os.makedirs(os.path.join("outputs/tests/stablediffusion", opt.model_name), exist_ok=True)
    if opt.analyze_tags:
        barWidth = 0.50
        keys = list(categories_errors.keys())
        fpc = [int(categories_errors[k][0]) for k in keys]
        fnc = [int(categories_errors[k][1]) for k in keys]
        r1 = np.arange(len(fpc))
        r2 = [x + barWidth for x in r1]
        fig = plt.figure()
        fig.set_figheight(14)
        fig.set_figwidth(14)
        plt.barh(r1, fpc, color='#557f2d', height=barWidth, edgecolor='white', label='false positive')
        plt.barh(r2, fnc, color='#7f6d5f', height=barWidth, edgecolor='white', label='false negative')
        plt.yticks([r + (barWidth - 0.25) for r in range(len(fpc))], keys)
        plt.ylabel('Categories', fontweight='bold')
        plt.xlabel('Errors', fontweight='bold')
        ax = plt.gca()
        ax.invert_yaxis() 
        plt.legend()
        plt.savefig(os.path.join("outputs/tests/stablediffusion", opt.model_name, "errors_per_category.png"))
        plt.close()

    captions_features = []
    
    bar = ChargingBar('EXTRACTING LINGUISTIC FEATURES', max=(len(captions_errors.keys())))

    nlp = en_core_web_sm.load()
    for caption in captions_errors.keys():
        features, col_names = get_text_features(caption, nlp)
        # -2: False positive; -1: False negative: +1: True positive: +2: True negative
        if captions_errors[caption][0] > 0: # Add a row with false positive value
            features[-2] = -2
            features[-1] = 1
        else: # Add a row with true positive
            features[-2] = 2  
            features[-1] = 0  
        captions_features.append(features)

        if captions_errors[caption][1] > 0: # Add a row with false negative value
            features[-2] = -1
            features[-1] = 1
        else: # Add a row with true negative
            features[-2] = 1
            features[-1] = 0
        
    
        captions_features.append(features)
        bar.next()
    bar.finish()


    df = pd.DataFrame(captions_features, columns=col_names)

    correlations = dict.fromkeys(col_names[:-2], 0)
    for column in col_names[:-2]:
        correlations[column] = df["LABEL"].corr(df[column])
    df_correlations = pd.DataFrame(correlations, index=["False Positive/Negative"])

    heatmap = sns.heatmap(df_correlations, annot=False, square=True)
    heatmap.invert_yaxis()
    heatmap.set(xlabel ="Linguistic Features", ylabel = "", title ='Correlation between false\n positive/false negative and linguistic features')
    fig = heatmap.get_figure()
    fig.savefig(os.path.join("outputs/tests/stablediffusion", opt.model_name, "correlation.png"))
    fig.clf()
    
    for column in col_names[:-2]:
        correlations[column] = df["BINARY_LABEL"].corr(df[column])
    df_correlations = pd.DataFrame(correlations, index=["Wrong/Correct Classification"])

    heatmap = sns.heatmap(df_correlations, annot=False, square=True)
    heatmap.invert_yaxis()
    heatmap.set(xlabel ="Linguistic Features", ylabel = "", title ='Correlation between wrong/correct\nclassification and linguistic features')
    fig = heatmap.get_figure()
    fig.savefig(os.path.join("outputs/tests/stablediffusion", opt.model_name, "correlation_binary_label.png"))