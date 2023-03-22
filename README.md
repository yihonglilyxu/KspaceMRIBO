# K-space Reconstruction of MR Images using Bayesian Methods
Yihong Xu, Chad W. Farris, Stephan W. Anderson, Xin Zhang, Keith A. Brown ... 

### Contact: Yihong Xu (yihongxu at bu dot edu)

--- 
# Abstract: 
> A central goal of modern magnetic resonance imaging (MRI) is to reduce the time required to produce high-quality images. Efforts have included hardware and software innovations including parallel imaging, compressed sensing, and deep learning-based reconstruction. Here, we propose and demonstrate a Bayesian method to build statistical libraries of MRI images in k-space and use these libraries to identify optimal subsampling paths and reconstruction processes. Specifically, we compute a multivariate normal distribution based upon Gaussian processes using a publicly available library of T1-weighted images of healthy brains. We combine this library with physics-informed envelope functions to only retain meaningful correlations in k-space. This covariance function is then used to guide a Bayesian optimization process to select a series of ring-shaped sub-sampling paths that optimally explore space while remaining practically realizable in commercial MRI systems. Combining optimized sub-sampling paths found for a range of images, we compute a generalized sampling path that, when used for novel images, produces superlative structural similarity and error in comparison to previously reported reconstruction processes (i.e. 96.3% structural similarity and <0.003 normalized mean squared error from sampling only 12.5% of the image). Finally, we use this reconstruction process without retraining on pathological data to show that reconstructed images are clinically useful for stroke identification.  

![alt text](https://github.com/yihonglilyxu/KspaceMRIBO/blob/main/KspaceMRIBO_pipeline.png)

## Dataset
 Download the IXI T1 dataset: 
  [https://paperswithcode.com/dataset/ixi-dataset](https://brain-development.org/ixi-dataset/)
##  Dictionaries and optimized sampling path 
 1. **Mean and Covariance Matrices:**
  Final multivariate normal distributions can be downloaded through [[link](https://drive.google.com/drive/folders/1aArnrAfU8tZ0KAci09W5le4NTeI4UnUn?usp=share_link)]. After downloading the dictionaries, please put them into [ `src/GPR_dictionary/ `](src/GPR_dictionary/) folder. 
 2. **Optimal Circular Sampling Path:**
[optimized circular sampling path for double develop function ](https://github.com/yihonglilyxu/KspaceMRIBO/blob/963f73001bd518aa722b47a180ebd4e5283fca13/src/Undersampled-Path/f2_6822_rpattern.npy)
## Methods Illustration 
This [jupyter notebook](https://github.com/yihonglilyxu/KspaceMRIBO/blob/82773ffaf6fba382b6812e6f1a9c22c190249bdb/Methods_illustration.ipynb)  or contains a simple tutorial
explaining how to preprocess the MR images, find the optimal sampling path and perform image reconstruction using BO methods. 

## Image reconstruction using your own MRI image  
Download the repository from Gihub. The main package is under the src folder. Please check the [Prerequisite](#Prerequisites) section for dependency requirements. 

Here we offer example input image data in [`src/MRI_data/`](src/MRI_data/) folder .

 - ***Option 1: Reconstruction of  a single 2D axial MRI image***

**input:** single MRI image in a 256*256 numpy array 

**example:**
 ```
%run recon_single_main.py --imgpath './MRI_data/example_img.npy'\

--undersampled-path './Undersampled-Path/f2_6822_rpattern.npy' \

--realcov-path './GPR_dictionary/realcov_L2_13.pt' \

--imagcov-path './GPR_dictionary/imagcov_L2_13.pt' \

--realmean-path './GPR_dictionary/train_realmean_6822.pt' \

--imagmean-path './GPR_dictionary/train_imagmean_6822.pt' \

--magmean-path './GPR_dictionary/train_magmean_6822.pt' \

--out-dir './MRI_data/'
```

 - ***Option 2: Test multiple 2D axial MRI images (Recommended!)***

 **input:** k-space MRI images in an h5 file, for each k-space image, the size should be (256,256,2). The third dimension is for real and imaginary part of k-space 

**example:**
 ```
%run recon_h5_main.py --imgpath './MRI_data/t1_test_example.h5' \

--undersampled-path './Undersampled-Path/f2_6822_rpattern.npy' \

--realcov-path './GPR_dictionary/realcov_L2_13.pt' \

--imagcov-path './GPR_dictionary/imagcov_L2_13.pt' \

--realmean-path './GPR_dictionary/train_realmean_6822.pt' \

--imagmean-path './GPR_dictionary/train_imagmean_6822.pt' \

--magmean-path './GPR_dictionary/train_magmean_6822.pt' \

--out-dir './MRI_data/'
```

 - **Outputs**: reconstruction NMSE, SSIM and PSNR values and reconstructed MRI images. 
 

### Prerequisites
```
torch~=1.11.0
numpy~=1.22.3
matplotlib~=3.4.1
h5py~=3.2.1
fastmri~=0.1.1
```
