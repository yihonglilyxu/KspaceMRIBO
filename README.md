# K-space Reconstruction of MR Images using Bayesian Methods
Yihong Xu, Keith A. Brown ... 

### Contact: Yihong Xu (yihongxu at bu dot edu)

--- 
# Abstract: 
> A central goal of modern magnetic resonance imaging (MRI) is to reduce the time required to produce high-quality images. Efforts have included hardware and software innovations including parallel imaging, compressed sensing, and deep learning-based reconstruction. Here, we propose and demonstrate a Bayesian method to build statistical libraries of MRI images in k-space and use these libraries to identify optimal subsampling paths and reconstruction processes. Specifically, we compute a multivariate normal distribution based upon Gaussian processes using a publicly available library of T1-weighted images of healthy brains. We combine this library with physics-informed envelope functions to only retain meaningful correlations in k-space. This covariance function is then used to guide a Bayesian optimization process to select a series of ring-shaped sub-sampling paths that optimally explore space while remaining practically realizable in commercial MRI systems. Combining optimized sub-sampling paths found for a range of images, we compute a generalized sampling path that, when used for novel images, produces superlative structural similarity and error in comparison to previously reported reconstruction processes (i.e. 96.3% structural similarity and <0.003 normalized mean squared error from sampling only 12.5% of the image). Finally, we use this reconstruction process without retraining on pathological data to show that reconstructed images are clinically useful for stroke identification.  

![alt text](https://github.com/yihonglilyxu/KspaceMRIBO/blob/main/KspaceMRIBO_pipeline.png)

## Dataset
* download the IXI T1 dataset: 
  [https://paperswithcode.com/dataset/ixi-dataset](https://brain-development.org/ixi-dataset/)
