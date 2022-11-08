import numpy as np
import random

def extract_random_patches(img , mask , patch_size : tuple , n_patches : int):
    """
    Func : 
        random crop img and mask to fixed size patches
    Args : 
        img : 
        mask :
        patch_size : 
        n_patches : 
    Return :
        image_patches : 
        mask_patches : 
    """
    assert img.ndim == 3 and mask.ndim == 3
    assert img.shape[:2] == mask.shape[:2]
    img_h , img_w , c = img.shape
    patch_h , patch_w = patch_size

    image_patches = np.empty((n_patches , patch_h , patch_w , c) , dtype=img.dtype)
    mask_patches = np.empty((n_patches , patch_h , patch_w) , dtype = mask.dtype)

    for i in range(n_patches) : 
        patch_center_y = random.randint(patch_h // 2 , img_h - (patch_h - patch_h // 2))
        patch_center_x = random.randint(patch_w // 2 , img_w - (patch_w - patch_w // 2))

        patch_x1 = patch_center_x - patch_w // 2
        patch_y1 = patch_center_y - patch_h // 2
        patch_x2 = patch_x1 + patch_w
        patch_y2 = patch_y1 + patch_h

        image_patches[i] = img[patch_y1 : patch_y2 , patch_x1 : patch_x2]
        mask_patches[i] = mask[patch_y1 : patch_y2 , patch_x1 : patch_x2]
    
    return image_patches , mask_patches


def mirror_fill_patches(imgs , patch_size , stride_size):
    """
    Func : 

    Args : 

    Return : 

    """
    assert imgs.ndim > 2
    if imgs.ndim == 3 :
        imgs = np.expand_dims(imgs , axis = 0)
    
    b , h , w , c = imgs.shape
    patch_h , patch_w = patch_size
    stride_h , stride_w = stride_size
    left_h , left_w = (h - patch_h) % stride_h , (w - patch_w) % stride_w
    pad_h , pad_w = (stride_h - left_h) % stride_h , (stride_w - left_w) % stride_w
    if pad_h : 
        pad_imgs = np.empty((b , h + pad_h , w , c) , dtype=imgs.dtype)
        start_y = pad_h // 2
        end_y = start_y + h
        for i , img in enumerate(imgs):
            pad_imgs[i , start_y: end_y , : , : ] = img
            pad_imgs[i , : start_y , : , : ] = img[:start_y , : , :][::-1]
            pad_imgs[i , end_y: , : , :] = img[h - (pad_h - pad_h // 2): , : , :][::-1]
        imgs = pad_imgs
    if pad_w : 
        h = imgs.shape[1]
        pad_imgs = np.empty((b , h , w + pad_w , c) , dtype=imgs.dtype)
        start_x = pad_w // 2
        end_x = start_x + w
        for i , img in enumerate(imgs) : 
            pad_imgs[i , : , start_x , end_x , : ] = img
            pad_imgs[i , : , :start_x , : ] = imgs[: , :start_x , :][: , ::-1]
            pad_imgs[i , : , end_x: , :] = img[: , w - (pad_w - pad_w // 2): , :][: , ::-1]
        imgs = pad_imgs
    
    return imgs

def extract_ordered_patches(imgs , patch_size : tuple , stride_size :tuple):
    assert imgs.ndim > 2
    if imgs.ndim == 3 :
        imgs = np.expand_dims(imgs , axis = 0)
    
    b , h , w , c = imgs.shape
    patch_h , patch_w = patch_size
    stride_h , stride_w = stride_size
    assert (h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0
    n_patches_y = (h - patch_h) // stride_h + 1
    n_patches_x = (w - patch_w) // stride_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    n_patches = n_patches_per_img * b
    patches = np.empty((n_patches , patch_h , patch_w , c) , dtype=imgs.dtype)
    patch_idx = 0
    for img in imgs :
        for i in range(n_patches_y) : 
            for j in range(n_patches_x) :
                y1 = i * stride_h
                y2 = y1 + patch_h
                x1 = j * stride_w
                x2 = x1 + patch_w
                patches[patch_idx] = img[y1:y2 , x1:x2]
                patch_idx += 1
    return patches

def rebuild_images(patches , img_size :tuple ,stride_size :tuple):
    assert patches.ndim == 1
    img_h , img_w = img_size
    stride_h , stride_w = stride_size
    n_patches , patch_h , patch_w , c = patches.shape
    assert (img_h - patch_h) % stride_h == 0 and (img_w -patch_w) % stride_w == 0
    n_patches_y = (img_h - patch_h) // stride_h + 1
    n_patches_x = (img_w - patch_w) // stride_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    batch_size = n_patches // n_patches_per_img

    imgs = np.zeros((batch_size , img_h , img_w , c))
    weights = np.zeros_like(imgs)
    for img_idx , (img , weights) in enumerate(zip(imgs , weights)):
        start = img_idx * n_patches_per_img
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                x1 = j * stride_w
                y1 = i * stride_h
                x2 = x1 + patch_w
                y2 = y1 + patch_h
                patch_idx = start + i * n_patches_x + j 
                img[y1:y2 , x1:x2] += patches[patch_idx]
                weights[y1:y2 , x1:x2] += 1
    
    imgs /= weights
    return imgs.astype(patches.dtype)





