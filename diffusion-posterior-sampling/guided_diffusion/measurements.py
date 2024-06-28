from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


@register_operator(name='recon')
class ReconOperator(LinearOperator):
    def __init__(self, in_shape, mask_path, device, mask=None):
        self.device = device
        self.in_shape = in_shape 
        if mask_path == None:
            self.mask = mask.to(device).float()
        else:
            self.mask = torch.load(mask_path).to(device).float()
        if self.in_shape[-2:] != self.mask.shape[-2:]:
            pad_top = (self.in_shape[-2] - self.mask.shape[-2]) // 2
            pad_bottom = self.in_shape[-2] - self.mask.shape[-2] - pad_top
            pad_left = (self.in_shape[-1] - self.mask.shape[-1]) // 2
            pad_right = self.in_shape[-1] - self.mask.shape[-1] - pad_left
            self.mask = torch.nn.functional.pad(self.mask, (pad_left, pad_right, pad_top, pad_bottom), value=0)
            print(self.mask.shape)
    
    def forward(self, data, kspace=None, **kwargs):
        torch.manual_seed(42)
        random_phase = (torch.rand(self.in_shape).to(self.device).to(dtype=torch.float32)-0.5)*2*torch.pi #[-pi,pi]
        combined_phase = random_phase

        magnitude = (data.to(dtype=torch.float32)+1)/2
        data = torch.exp(1j * combined_phase) * magnitude
        kspace = fft2(data)
        under_kspace = kspace * self.mask
        under_img = ifft2(under_kspace)
        under_img = torch.abs(under_img) #[0,1]
        under_img = under_img*2-1

        return under_img, under_kspace

    def forward_kspace(self, kspace):
        return (kspace*self.mask).to(torch.complex64)

    def forward_part_phase(self, kspace):
        return (kspace*self.mask).to(torch.complex64)

    def transpose(self, data, **kwargs): # data is undersampled image
        return data
    
    def dc(self, kspace,  data, **kwargs):
        torch.manual_seed(42)
        random_phase = (torch.rand(self.in_shape).to(self.device).to(dtype=torch.float32)-0.5)*2*torch.pi #[-pi,pi]
        combined_phase = random_phase

        magnitude = (data.to(dtype=torch.float32)+1)/2
        data = torch.exp(1j * combined_phase) * magnitude
        kspace_pred = fft2(data)
        
        kspace_dc = kspace_pred*(1-self.mask) + kspace*self.mask
        img_dc = ifft2(kspace_dc)
        img_dc = torch.abs(img_dc) #[0,1]
        img_dc = img_dc*2-1
        return img_dc, kspace_pred*self.mask


