import os
choose_mask = '../../mask/generated_masks/mask_4x_uniform1d.pth'

from inverse import *
from torch.utils.data import Dataset, Subset, DataLoader

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_false',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=1000,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--dpm_solver",
    action='store_true',
    help="use dpm_solver sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=320,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=320,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="../../stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
## 
parser.add_argument(
    "--dps_path",
    type=str,
    default='../../diffusion-posterior-sampling/',
    help="DPS codebase path",
)
parser.add_argument(
    "--task_config",
    type=str,
    default='configs/super_resolution_config_psld.yaml',
    help="task config yml file",
)
parser.add_argument(
    "--diffusion_config",
    type=str,
    default='configs/diffusion_config.yaml',
    help="diffusion config yml file",
)
parser.add_argument(
    "--model_config",
    type=str,
    default='configs/model_config.yaml',
    help="model config yml file",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1e-1,
    help="inpainting error",
)
parser.add_argument(
    "--omega",
    type=float,
    default=1,
    help="measurement error",
)
parser.add_argument(
    "--inpainting",
    type=int,
    default=0,
    help="inpainting",
)
parser.add_argument(
    "--general_inverse",
    type=int,
    default=1,
    help="general inverse",
)
parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help='input image',
)
parser.add_argument(
    "--skip_low_res",
    action='store_true',
    help='downsample result to 256',
)
parser.add_argument(
    "--ffhq256",
    action='store_true',
    help='load SD weights trained on FFHQ',
)

opt = parser.parse_args()

seed_everything(opt.seed)

config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)



batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))



sys.path.append(opt.dps_path)

import yaml
from guided_diffusion.measurements import get_noise, get_operator
from util.img_utils import clear_color, mask_generator
import torch.nn.functional as f
import matplotlib.pyplot as plt


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


model_config=opt.dps_path+opt.model_config
diffusion_config=opt.dps_path+opt.diffusion_config
task_config=opt.dps_path+opt.task_config


model_config = load_yaml(model_config)
diffusion_config = load_yaml(diffusion_config)
task_config = load_yaml(task_config)


from PIL import Image
from math import pi
import pickle
import os
import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from einops import rearrange, repeat
import torch.distributions as distributions
import cv2

def psnr(img1, img2, max_pixel=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))

class MaskFunc_Cartesian:
    def __init__(self, center_fractions, accelerations):

        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        self.rng.seed(seed)
        num_cols = shape[-1]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs + 1e-10)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        mask_shape = [1 for _ in shape] 
        mask_shape[-1] = num_cols
        mask = mask.reshape(*mask_shape).astype(np.float32)
        mask = np.repeat(mask, shape[0], axis=0) 
        return mask

class FastMRI_knee_magnitude(Dataset):
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths
        self.file_list = []
        for ind, folder_path in enumerate(folder_paths):
            self.file_list += [file for file in os.listdir(folder_path) if file.endswith('.pt') ]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        img_prior1 = pickle.load(open(os.path.join(self.folder_paths[0], file_name), 'rb'))['img']
        magnitude = np.abs(img_prior1)/np.abs(img_prior1).max() #[0,1]
        phase = (np.angle(img_prior1)-np.angle(img_prior1).min())/(np.angle(img_prior1).max()-np.angle(img_prior1).min())
        normalized_data = np.exp(1j*phase)*magnitude

        kspace = fft2(torch.from_numpy(normalized_data))
        kspace = torch.stack([kspace,kspace,kspace], dim=0).to(torch.complex64)
        
        magnitude = magnitude*2-1 #[-1,1]
        magnitude = magnitude.astype(np.float32)
        magnitude = np.stack([magnitude, magnitude, magnitude]).astype(np.float32)
        return magnitude, kspace
    

dset_test = FastMRI_knee_magnitude(folder_paths=["../../data/example/val"])
test_dataloader = DataLoader(dset_test, batch_size=1, shuffle=False)


import guided_diffusion
import importlib
importlib.reload(guided_diffusion)
import guided_diffusion
from guided_diffusion.measurements import get_operator

import sys
sys.path.append(opt.dps_path)

import yaml
from util.img_utils import clear_color, mask_generator
import torch.nn.functional as f
import matplotlib.pyplot as plt


measure_config = {'operator': {'name': 'recon', 'in_shape': (1, 3, 320, 320), 'mask_path': choose_mask}, 
                'noise': {'name': 'gaussian', 'sigma': 0}}
operator = get_operator(device=device, **measure_config['operator'])


from skimage.metrics import structural_similarity as ssim

def psnr(img1, img2, max_pixel=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Prepare Operator and noise
measure_config = task_config['measurement']
noiser = get_noise(**measure_config['noise'])


def psnr(img1, img2, max_pixel=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

from mrsampler import DDIMSampler
sampler = DDIMSampler(model)

for img in test_dataloader:
    org_image = torch.clone(img[0][0])
    org_image = org_image[None,:,:,:].cuda()

    kspace_ = fft2((img[0]+1)/2).cuda()
    kspace_measurement = operator.forward_kspace(kspace_) # original, no corruption
    y, kspace = operator.forward(org_image, kspace_measurement) # kspace is with corruption 
    
    y_n = y
    mask = None

    forward_image = y.permute(0,2,3,1).cpu().numpy()[0]
    original_image = org_image.permute(0,2,3,1).cpu().numpy()[0]
    y_n_image = y_n.permute(0,2,3,1).cpu().numpy()[0]

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None

                    # pdb.set_trace()
                    if opt.scale != 1.0 :
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, intermediate = sampler.sample_fast(
                                                    eta=0.05,
                                                    S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    x_T=start_code,
                                                    ip_mask = mask,
                                                    measurements = y_n,
                                                    operator = operator,
                                                    gamma = opt.gamma,
                                                    inpainting = opt.inpainting,
                                                    omega = opt.omega,
                                                    log_every_t=1,
                                                    noiser=noiser, 
                                                    bicubic_image=y,
                                                    device=device, 
                                                    kspace=kspace,
                                                    general_inverse=7)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    pred_image = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()                    

                    break


    original_img = np.mean((original_image + 1)/2, axis=-1)
    pred_img = np.mean(pred_image[0], axis=-1)


    psnr_recon = psnr(original_img, pred_img)
    ssim_recon, _ = ssim(original_img.astype('float64'), pred_img.astype('float64'), full=True)
    mse_recon = np.abs((original_img-pred_img)).sum()/(320*320)
    print("Enjoy.", psnr_recon, ssim_recon, mse_recon)