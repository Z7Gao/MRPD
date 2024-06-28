# README
Here, we provide the PTH files for all masks used in our evaluation. We also show the masks generation process, which is built on the utils.py from the official Score-MRI (https://github.com/HJ-harry/score-MRI/blob/main/utils.py) and FastMRI (https://github.com/facebookresearch/fastMRI/tree/main) codebases.
## File Structure
```
.
├── generated_masks
│   ├── mask_12x_uniform1d_256.pth
│   ├── mask_12x_uniform1d.pth
│   ├── mask_15x_poisson_256.pth
│   ├── mask_15x_poisson.pth
│   ├── mask_4x_gaussian1d_256.pth
│   ├── mask_4x_gaussian1d.pth
│   ├── mask_4x_gaussian2d_256.pth
│   ├── mask_4x_gaussian2d.pth
│   ├── mask_4x_uniform1d_256.pth
│   ├── mask_4x_uniform1d.pth
│   ├── mask_8x_gaussian1d_256.pth
│   ├── mask_8x_gaussian1d.pth
│   ├── mask_8x_gaussian2d.pth
│   ├── mask_8x_guassian2d_256.pth
│   ├── mask_8x_poisson_256.pth
│   ├── mask_8x_poisson.pth
│   ├── mask_8x_uniform1d_256.pth
│   └── mask_8x_uniform1d.pth
├── mask_gen.ipynb
├── README.md
└── utils.py
```
## all_vis.pdf & Folder: generated_masks
While part of the masks has been shown in the paper (Figure 3,5,6), we provide the complete PTH files in **generated_masks** for all sampling masks used in our paper here to enhance our reproducibility. For better readability, we provide the **all_vis.pdf** that contains a layout for all masks. 

Specifically, we use four different sampling patterns, each with different sampling rates:
<ul>
    <li> Uniform 1D (acc_rate = 4, 8, 12, with the fully-sampled central regions including 8% of all k-space lines). </li>
    <li> Gaussian 1D (acc_rate = 4, 8, with the fully-sampled central regionS including 8% and 4% of all k-space lines, respectively). </li>
    <li> Gaussian 2D (acc_rate = 4, 8)</li>
    <li> VD Poisson Disks (acc_rate = 8, 15)</li>
</ul>

Notice that different masks are required for IXI and FastMRI, due to the differences in image size. The files with a "_256.pth" ending are for IXI and the rest are for FastMRI, which consists of complexed-value images of shape (320,320).

## Code: mask_gen_vis.ipynb
We demonstrate our mask-generating process and how to visualize the pth files in the folder.

Your generated masks may differ with ours with the same set of (acc_rate, sampling_pattern) due to the randomness of sampling, and we evaluate all methods with those provided in generated_masks.