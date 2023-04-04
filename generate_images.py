#from stable_diffusion.scripts.txt2img_custom import txt2img as sd
from glide import txt2img as gl
import json
import argparse
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
import os
#from pytorch_lightning import seed_everything
import shutil
import random
import pandas as pd 
import requests
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=100, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--n_samples', default=4, type=int,
                        help='Number of generated images.')
    parser.add_argument('--considered_images', default=6000, type=int,
                        help='Number of considered images.')
    parser.add_argument('--dataset', default=0, type=int,
                        help='Dataset (0: COCO; 1: Wikimedia).')
    parser.add_argument('--generator', default=0, type=int,
                        help='Generator (0: Stable Diffusion; 1: GLIDE).')
    parser.add_argument('--list_file', default="../datasets/coco/annotations/captions_val2014.json", type=str,
                        help='List of images.')
    parser.add_argument('--copy_files', default=False, action="store_true",
                        help='Do files copy')
    parser.add_argument('--data_path', default="../datasets/coco/val2014", type=str,
                        help='Path to data.')
    parser.add_argument('--force_captions', default="../datasets/diffused_wikipedia/test", type=str,
                        help='Consider the captions from a specific folder.')
    parser.add_argument('--excluded_images', default=["../datasets/diffused_wikipedia/train", "../datasets/diffused_wikipedia/val"], type=list,
                        help='Path to excluded images.')
    parser.add_argument('--output_path', default="../datasets/glide_diffused_wikipedia/test", type=str,
                        help='Output path.')
    opt = parser.parse_args()
    #seed_everything(42)
    random.seed(42)
    if opt.force_captions == "":
        if opt.dataset == 0:
            f = open(opt.list_file)
            data = json.load(f)
            images = {}
            counter = 0
            for element in data["images"]:
                element = dict(element)
                id = element["id"]
                if id not in images:
                    images[id] = os.path.join(opt.data_path, element["file_name"])

            captioned_images = {}
            for element in data["annotations"]:
                element = dict(element)
                id = element["image_id"]
                if id in images:
                    captioned_images[id] = [images[id], element["caption"]]
                
            
            captioned_images = list(captioned_images.items())
            captioned_images = sorted(captioned_images, key=lambda k: random.random())
            captioned_images = captioned_images[:opt.considered_images]
            prompts = [row[1][1] for row in captioned_images]
            
            if opt.copy_files:
                for row in captioned_images:
                    src_image = row[1][0]
                    dst_path = os.path.join(opt.output_path, row[1][1])
                    os.makedirs(dst_path, exist_ok = True)
                    dst_image = os.path.join(dst_path, "real.png")        
                    shutil.copy(src_image, dst_image)
            
        else:
            excluded_images = []
            if len(opt.excluded_images) > 0:
                for excluded in opt.excluded_images:
                    excluded_images.extend(os.listdir(excluded))
            

            df = pd.read_csv(opt.list_file, sep=',')
            indexes_to_drop = []
            for index, row in df.iterrows():
                filename, file_extension = os.path.splitext(row['image_url'])
                tags = ast.literal_eval(row["page_tags"])
                if row['caption'] in excluded_images or len(row['caption']) > 77 or len(row['caption']) < 10 or ("png" not in file_extension and "jpg" not in file_extension) or len(tags) == 0:
                    indexes_to_drop.append(index)
            df = df.drop(indexes_to_drop)
            df = df.sample(frac=1).reset_index(drop=True)
            df = df.head(opt.considered_images)
            prompts = []
            rows = len(df)
            for index, row in df.iterrows():
                url = row['image_url']
                caption = row['caption']
                tags = ast.literal_eval(row["page_tags"])
                if "/" in caption:
                    caption = caption.replace("/", "-")

                prompts.append(caption)
                
                if not opt.copy_files:
                    continue

                headers = {
                    'User-Agent': 'My User Agent 1.0',
                    'From': 'davidecoccomini@edge-nd1.isti.cnr.it'  
                }
                response = requests.get(url, headers=headers)
                filename, file_extension = os.path.splitext(url)
                dst_path = os.path.join(opt.output_path, caption)
                os.makedirs(dst_path, exist_ok = True)
                output_file = open(os.path.join(dst_path, "real" + file_extension), 'wb')
                output_file.write(response.content)
                output_file.close()

                with open(os.path.join(dst_path, "tags.txt"), 'w+') as ft:
                    for tag in tags:
                        tag = tag.replace("http://dbpedia.org/ontology/", "")
                        ft.write(f"{tag}\n")

                if index % 100 == 0 and index > 0:
                    print("Saved", index, "/", rows ,"images")
    else:
        prompts = sorted(os.listdir(opt.force_captions))
        if opt.copy_files:
            rows = len(prompts)
            for index, prompt in enumerate(prompts):
                src_path = os.path.join(opt.force_captions, prompt)
                dst_path = os.path.join(opt.output_path, prompt)
                os.makedirs(dst_path, exist_ok = True)

                for image in os.listdir(src_path):
                    if "real" in image:
                        image_name = os.path.basename(image)
                        break
                src_image = os.path.join(src_path, image_name)
                dst_image = os.path.join(dst_path, image_name)
                shutil.copy(src_image, dst_image)

                
                if index % 100 == 0 and index > 0:
                    print("Saved", index, "/", rows ,"images")

    if opt.generator == 0:
        txt2img = sd
    else:
        txt2img = gl
                        

    generated_images = txt2img(prompts=prompts, skip_grid=True, skip_save=False, n_samples=opt.n_samples, outdir=opt.output_path)

    

