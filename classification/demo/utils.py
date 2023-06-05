import os , os.path as osp
import json
import random
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

SUPPORTED = [".jpg" , ".jpeg" , ".png"]

def conversion_type(data):
    if isinstance(data , bytes):
        return str(data , encoding='utf-8')
    if isinstance(data , dict):
        return dict(data)
    return json.JSONEncoder.default(data)

def read_split_data(data_root , val_rate=0.2 , plot=False):
    random.seed(0)
    assert osp.exists(data_root) , \
        f"Dataset root : {data_root} does not exit."
    
    class_list = [cla for cla in os.listdir(data_root) if osp.isdir(osp.join(data_root , cla))]
    class_list.sort()
    class_indices = dict((cls , index) for index , cls in enumerate(class_list))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('classes.json' , 'w') as json_file:
        json_file.write(json_str)
        
    train_img_path = []
    train_ann_path = []
    val_img_path = []
    val_ann_path = []
    class_sample_num = []
    
    for cls in class_list:
        cls_path = osp.join(data_root , cls)
        image_lists = [osp.join(data_root , cls , index) for index in os.listdir(cls_path) \
                       if osp.splitext(index)[-1].lower() in SUPPORTED]
        image_lists.sort()
        image_class = class_indices[cls]
        class_sample_num.append(len(image_lists))
        # Proportionately random sampling of validation samples across classes
        val_path = random.sample(image_lists , k=int(len(image_lists) * val_rate))
        
        for image_path in image_lists:
            if image_path in val_path:
                val_img_path.append(image_path)
                val_ann_path.append(image_class)
            else:
                train_img_path.append(image_path)
                train_ann_path.append(image_class)
    
    print(f"{sum(class_sample_num)} images were found in the dataset.")
    print(f"{len(train_img_path)} images for training.")
    print(f"{len(val_img_path)} images for validation.")
    assert len(train_img_path) > 0 , \
        f"The number of training must greater than 0."
    assert len(val_img_path) > 0 , \
        f"The number of training must greater than 0."
    
    if plot:
        plt.bar(range(len(class_list)) , class_sample_num, align='center')
        plt.xticks(range(len(class_list)) , class_list)
        for index , value in enumerate(class_sample_num):
            plt.text(x=index , y=value+5 , s=str(value) , ha='center')
        plt.xlabel("Image classes")
        plt.ylabel("Number")
        plt.title('Data class distribution')
        plt.show()
    
    return train_img_path , train_ann_path , val_img_path , val_ann_path
        
        
        
    
        
        
            
        
        
    
    
    
    
    