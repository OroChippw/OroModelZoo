import sys
import os , os.path as osp
import argparse
import math
import tqdm

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import read_split_data
from datasets import SampleDataSet as DataSet
from ..models import VisionTransformer_base_patch16_r224_in21k as VisionTransformer

def train(model , data_loader , optimizer , loss_function , device , epoch):
    model.train()
    loss_total = torch.zero(1).to(device)
    acc_num = torch.zero(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader , file=sys.stdout)
    for step , data in enumerate(data_loader):
        images , labels = data
        sample_num += images.shape[0]
        
        pred = model(images.to(device))
        pred_classes = torch.max(pred , dim=1)[1]
        acc_num += torch.eq(pred_classes , labels.to(device)).sum()
        
        loss = loss_function(pred , labels.to(device))
        loss.backward()
        loss_total += loss.detach()
        
        data_loader.desc = f"[train epoch {epoch}] loss : {loss_total.item() / (step + 1)} , acc : {acc_num.item() / sample_num}"
        
        optimizer.step()
        optimizer.zero_grad()
    return loss_total.item() / (step + 1) , acc_num.item() / sample_num

@torch.no_grad()
def evaluate(model , data_loader , loss_function , device , epoch):
    model.eval()
    loss_total = torch.zero(1).to(device)
    acc_num = torch.zero(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader , file=sys.stdout)
    for step , data in enumerate(data_loader):
        images , labels = data
        sample_num += images.shape[0]
        
        pred = model(images.to(device))
        pred_classes = torch.max(pred , dim=1)[1]
        acc_num += torch.eq(pred_classes , labels.to(device)).sum()
        
        loss = loss_function(pred , labels.to(device))
        loss_total += loss
        
        data_loader.desc = f"[vaild epoch {epoch}] loss : {loss_total.item() / (step + 1)} , acc : {acc_num.item() / sample_num}"
    
    return loss_total.item() / (step + 1) , acc_num.item() / sample_num
        
    

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not osp.exists('./output'):
        os.makedirs('./output')
    
    tensorboard_writer = SummaryWriter()
    
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
    val_dataloader = torch.utils.data.DataLoader(val_dataset , batch_size=batch_size , 
                                                    shuffle=False , pin_memory=True , collate_fn=train_dataset.collate_fn)
    
    model = VisionTransformer(num_classes=args.num_classes).to(device)
    
    if args.weight is not None:
        assert osp.exists(args.weight) , f"File {args.weight} is not exists"
        # TODO
        weight_dict = torch.load(args.weight , map_location=device)
        del_key = ['head.weight' , 'head.bias'] if model.has_logit \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_key:
            del weight_dict[k]
        
        model.load_state_dict(weight_dict , strict=False)
        
    if args.freeze_layers is not None:
        for name , para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg , lr = args.lr , momentum=0.9 , weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer , T_max=args.epoches)
    loss_function = torch.nn.CrossEntropyLoss()
    tensorboard_tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    
    for epoch in range(args.epoches):
        train_loss , train_acc = train(model , train_dataloader , optimizer , loss_function , device , epoch)        
        scheduler.step()
        val_loss , val_acc = evaluate(model , val_dataloader , loss_function , device , epoch)
           
        tensorboard_writer.add_scalar(tensorboard_tags[0] , train_loss , epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[1] , train_acc , epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[2] , val_loss , epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[3] , val_acc , epoch)
        tensorboard_writer.add_scalar(tensorboard_tags[4] , optimizer.param_groups[0]["lr"] , epoch)
        
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
          

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
     
     
    
    
    