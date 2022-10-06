import logging
import torch
import os.path as osp
import argparse
from PIL import Image
import torchvision.transforms as transforms

from models import MobileNetv2

logging.getLogger().setLevel(logging.INFO)

def args_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--img_path",
        type=str,
        help="the image be inferenced"
    )
    parse.add_argument(
        "--ckpt_path",
        type=str,
        help="the weight used to inference"
    )


    args = parse.parse_args()
    return args

def main():
    args = args_parse()

    assert osp.exists(args.img_path) , \
        'file {} does not exist.'.format(args.img_path)
    assert osp.exists(args.ckpt_path), \
        "ckpt {} does not exist.".format(args.ckpt_path)

    file_path = args.img_path
    ckpt_path = args.ckpt_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    img_ = Image.open(file_path).copy()
    img_ =predict_transform(img_)
    img_ = torch.unsqueeze(img_ , dim = 0)

    classlist_ = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print(classlist_)

    model = MobileNetv2(num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    with torch.no_grad():
        output_ = model(img_.to(device))
        output_ = torch.squeeze(output_).cpu()
        print(output_)
        pred_ = torch.softmax(output_ , dim=0)
        print(pred_)
        result_ = torch.argmax(pred_).numpy()
        print(result_)

    logging.info("class : {} prob : {}".format(classlist_[result_] , pred_[result_].numpy()))
    print("#-----------#")
    for i in range(len(pred_)):
        logging.info("class : {} prob : {}".format(classlist_[i], pred_[i].numpy()))


if __name__ == '__main__':
    main()