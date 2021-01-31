import sys, os, glob
import torch

def _is_tensor_a_torch_image(x: torch.Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img):
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")

def _blend(img1: torch.Tensor, img2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

def adjust_contrast(img: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    ''' 
    Adjust contrast of image with functionality identical to TF.adjust_contrast,
    although works for images with arbitrary number of channels.
    
    Args
    ----
    img (torch.Tensor): the image to have contrast adjusted

    contrast_factor (float): level of contrast adjustment

    Outputs
    -------
    adjusted_img (torch.Tensor): the input image with contrast adjusted by 
    contrast_factor 
    '''

    if contrast_factor < 0:
        raise ValueError(f'contrast factor ({contrast_factor}) is non-negative.')

    _assert_image_tensor(img)
    
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    mean = torch.mean(img.to(dtype), dim=(-2,-1), keepdim=True)

    return _blend(img, mean, contrast_factor)

    

       