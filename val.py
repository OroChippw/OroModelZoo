import torch
import time

@torch.no_grad
def model_val(model , dataloader , epoch , device):
    start = time.time()
    model.val()
    all_acc = 0
    for idx , (img , labels) in enumerate(dataloader):
        img = img.to(device)
        labels = labels.to(device)
        pred_ = model(img)

        cur_acc = (pred_.data.max(dim = 1)[1] == labels).sum() / len(labels)
        all_acc += cur_acc

    end_time = time.time()
    print("epoch:{} acc:{}".format(epoch,all_acc*100/len(dataloader)))
    return all_acc/len(dataloader)
