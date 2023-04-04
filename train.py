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
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from albumentations import Compose, PadIfNeeded, CenterCrop
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import collections
from images_dataset import ImagesDataset
from progress.bar import ChargingBar
from utils import check_correct, resize, get_n_params, center_crop
#from transformers import AutoImageProcessor, SwinModel
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights

IMAGE_SIZE = 224

def read_images(caption, dataset, data_path, mode, transform):
    caption_path = os.path.join(data_path, caption)
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

        image = center_crop(image)
        image = transform(image=image)['image']
        row = (image, caption, label)
        dataset.append(row)

def create_pre_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT)
    ])


# Main body
if __name__ == "__main__":
    seed_everything(42)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=100, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--training_path', default='../datasets/diffused_coco/train', type=str,
                        help='Images directory')
    parser.add_argument('--validation_path', default='../datasets/diffused_coco/val', type=str,
                        help='Validation directory')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--model_path', type=str, default='outputs/models/coco',
                        help='Path to save checkpoints.')
    parser.add_argument('--gpu_id', default=3, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--config', type=str, default='',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: SwinViT, 1: Resnet50, 2: CLIP+MLP; 3: CLIP+LMLP)")
    parser.add_argument('--clip_mode', type=int, default=0, 
                        help="Which model architecture version to be trained (0: Resnet50, 1: ViT)")
    parser.add_argument('--mode', type=int, default=0, 
                        help="Which mode to be used (0: Image-Only, 1: Image+Text[Only CLIP])")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--show_stats', type=bool, default=True, 
                        help="Show stats")
    parser.add_argument('--logger_name', default='runs/train/coco_large',
                        help='Path to save the model and Tensorboard log.')
                        
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
            clip_model, preprocess = clip.load("RN50", device=torch.device('cpu'))
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
            
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model parameters:", params)

    # Read dataset
    training_captions = os.listdir(opt.training_path)
    mgr = Manager()
    train_dataset = mgr.list()
    transform = create_pre_transform(IMAGE_SIZE)

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(training_captions)) as pbar:
            for v in p.imap_unordered(partial(read_images, dataset=train_dataset, data_path = opt.training_path, mode = opt.mode, transform=transform), training_captions):
                pbar.update()
    
    validation_captions = os.listdir(opt.validation_path)
    validation_dataset = mgr.list()

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(validation_captions)) as pbar:
            for v in p.imap_unordered(partial(read_images, dataset=validation_dataset, data_path = opt.validation_path, mode = opt.mode, transform=transform), validation_captions):
                pbar.update()
    
    train_labels = [float(row[2]) for row in train_dataset]
    train_captions = [row[1] for row in train_dataset]
    train_dataset = [row[0] for row in train_dataset]

    validation_labels = [float(row[2]) for row in validation_dataset]
    validation_captions = [row[1] for row in validation_dataset]
    validation_dataset = [row[0] for row in validation_dataset]

    train_samples = len(train_dataset)
    validation_samples = len(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(train_labels)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(validation_labels)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    if opt.config != '':
        batch_size = config['training']['bs']
    else:
        batch_size = 8

    train_dataset = ImagesDataset(np.asarray(train_dataset), train_captions, np.asarray(train_labels), IMAGE_SIZE)
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = ImagesDataset(np.asarray(validation_dataset), validation_captions, np.asarray(validation_labels), IMAGE_SIZE, mode='validation')
    val_dataset = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    
    
    # TRAINING
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()
    
    model.train()   
    #if opt.model == 0:
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    #else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    num_steps = int(opt.num_epochs * len(dl))

    lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config['training']['lr'] * 1e-3,
                cycle_limit=9,
                t_in_epochs=False,
    )
    starting_epoch = 0
    if  opt.resume != '':
        model_weights = opt.resume
        epoch = int(model_weights.split("_")[-1])
        while not os.path.exists(model_weights):
            epoch = int(model_weights.split("_")[-1])
            model_name = '_'.join(model_weights.split("_")[:-1])
            model_weights = model_name + "_" + str(epoch - 1)
            print("Trying new model weights", model_weights)
            if epoch == 0:
                print("No model found.")
                exit()
        starting_epoch = epoch + 1
        model.load_state_dict(torch.load(model_weights))
        print("Weights loaded")
    else:
        print("No checkpoint loaded.")
        print(opt.resume)
        
    model = model.to(opt.gpu_id)
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        save_model = False
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*batch_size)+len(val_dataset))
        train_correct = 0
        positive = 0
        negative = 0
        
        train_batches = len(dl)
        val_batches = len(val_dataset)
        total_batches = train_batches + val_batches

        for index, (images, captions, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.to(opt.gpu_id)
            captions = captions.to(opt.gpu_id)
            if opt.model < 2 or opt.model == 7:
                y_pred = model(images)
            else:
                with torch.no_grad():
                    image_features = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    if opt.mode == 1:
                        text_features = clip_model.encode_text(captions)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        features = torch.cat((image_features, text_features), dim = 1)
                    else:
                        features = image_features
                
                features = features.float()

                #features = torch.nn.functional.normalize(features)
                y_pred = model(features)
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step_update((t * (train_batches) + index))
            counter += 1
            total_loss += round(loss.item(), 2)
            for i in range(batch_size):
                bar.next()

            if index%100 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*batch_size), "Train 0s: ", negative, "Train 1s:", positive)  


        
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
       
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_captions, val_labels) in enumerate(val_dataset):
    
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            val_images = val_images.to(opt.gpu_id)
            val_captions = val_captions.to(opt.gpu_id)

            val_labels = val_labels.unsqueeze(1)
            with torch.no_grad():
                if opt.model < 2 or opt.model == 7:
                    val_pred = model(val_images)
                else:
                    image_features = clip_model.encode_image(val_images)
                    if opt.mode == 1:
                        text_features = clip_model.encode_text(val_captions)
                        features = torch.cat((image_features, text_features), dim=1)
                        features = torch.cat((image_features, text_features), dim = 1)
                    else:
                        features = image_features
                    features = features.float()
                    features = torch.nn.functional.normalize(features)
                    val_pred = model(features)
                val_pred = val_pred.cpu()
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                val_counter += 1
                bar.next()
            
        #scheduler.step()
        bar.finish()
        

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            save_model = True
            not_improved_loss = 0
        
        tb_logger.add_scalar("Training/Accuracy", train_correct, t)
        tb_logger.add_scalar("Training/Loss", total_loss, t)
        tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], t)
        tb_logger.add_scalar("Validation/Loss", total_loss, t)
        tb_logger.add_scalar("Validation/Accuracy", val_correct, t)

        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]))
    
        
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        if save_model and t > opt.num_epochs-20:
            torch.save(model.state_dict(), os.path.join(opt.model_path, opt.model_name + "_" + str(t)))

    #training_set = list(dict.fromkeys([os.path.join(opt.data_path, os.path.dirname(row[0].split(" "))) for row in training_set]))
  
