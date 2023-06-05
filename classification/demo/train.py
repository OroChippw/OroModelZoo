import os , os.path as osp
import argparse

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import read_split_data
from datasets import SampleDataSet as DataSet
from ..models import vision_transformer

def main(args):
    device_ = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not osp.exists('./output'):
        os.makedirs('./output')
    
    # writer_ = SummaryWriter()
    
    train_img_path , train_ann_path , val_img_path , val_ann_path = read_split_data(args.data_path , plot=False)
    
    data_transform = {
        "train" : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5])
        ]),
        "val" : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5] , [0.5,0.5,0.5])
        ])
    }
    
    train_dataset = DataSet(image_path=train_img_path , 
                            image_classes=train_ann_path,
                            transform=data_transform["train"])
    val_dataset = DataSet(image_path=val_img_path,
                          image_classes=val_ann_path,
                          transform=data_transform["val"])
    
    batch_size = args.batch_size
    num_workers = min([os.cpu_count() , batch_size if batch_size > 1 else 0 , 8])
    print(f"Using {num_workers} num_workers each process")
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset , batch_size=batch_size , 
                                                   shuffle=True , pin_memory=True , 
                                                   num_workers=num_workers , collate_fn=train_dataset.collate_fn)
    
    
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes' , type=int , default=10 , required=True)
    parser.add_argument('--epoches' , type=int , default=100 , required=True)
    parser.add_argument('--batch-size' , type=int , default=10 , required=True)
    parser.add_argument('--lr' , type=float , default=0.001)
    
    parser.add_argument('--data-path' , type=str , default=10 , required=True)
    parser.add_argument('--weights' , type=str , required=True)
    
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device' , default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    
    opt = parser.parse_args()
    
    main(opt)
     
     
    
    
    