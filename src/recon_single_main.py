import argparse
import pathlib

import torch
import os
import numpy as np
import matplotlib as m
import matplotlib.pyplot as plt


import fastmri 
from fastmri.fftc import fft2c_new, ifft2c_new
from fastmri.math import complex_abs


from loss import mse,nmse,psnr,ssim
from img_recon import bayesian_mean_samp_nor
from mask import rdic, mask_of_r_order

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {} 

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--imgsz",default = 256)
    parser.add_argument("--w",default = 48)
    
    parser.add_argument("--imgpath",type=str,help="The path of the input real space MRI image", required=True)
    parser.add_argument("--undersampled-path",type=str,help="The path of k-space undersampling pattern", required=True)
    
    parser.add_argument("--realcov-path",type=str,help="The path of covariance matrix of real part of k-space", required=True)
    parser.add_argument("--imagcov-path",type=str,help="The path of covariance matrix of imaginary part of k-space", required=True)
    parser.add_argument("--realmean-path",type=str,help="The path of mean value of real part of k-space", required=True)
    parser.add_argument("--imagmean-path",type=str,help="The path of mean value of imaginary part of k-space", required=True)
    parser.add_argument("--magmean-path",type=str,help="The path of mean value of k-space magnitude", required=True)
    
    parser.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path("output"), help='Name of the directory to store the output')
   

    return parser.parse_args()

args = get_args()
imgsz = args.imgsz
w = args.w
#load dictionaries 
realcov = torch.load(args.realcov_path)
imagcov = torch.load(args.imagcov_path)

realmean = torch.load(args.realmean_path).reshape((imgsz-2*w,imgsz-2*w))
imagmean = torch.load(args.imagmean_path).reshape((imgsz-2*w,imgsz-2*w))

realvar= torch.diagonal(realcov)
imagvar= torch.diagonal(imagcov)

magmean = torch.load(args.magmean_path).reshape((imgsz-2*w,imgsz-2*w))

realmean = realmean/magmean
imagmean = imagmean/magmean

#load single 2D MRI image and transform it to k-space
im = np.load(args.imgpath)
im /= np.max(im)
im = im.reshape((imgsz,imgsz,1))
im_k = torch.from_numpy(np.concatenate((im,np.zeros_like(im)),2)).float()
#transform the real space image to k-space image 
im_k = fft2c_new(im_k)
#load the undersampling pattern
rpattern = np.load(args.undersampled_path)
r_dic = rdic(imgsz,w)
mask_r,mask_r_ratio= mask_of_r_order(imgsz-2*w,rpattern,r_dic)

lamb = 1e-3
ind = np.linspace(0,(imgsz-2*w)**2-1,(imgsz-2*w)**2,dtype = int)
rcon,nmse,ssim,psnr = bayesian_mean_samp_nor(imgsz,w,im_k,ind[mask_r.view(-1) == 1],realcov,imagcov,realmean,imagmean,magmean,lamb)

#print reconstruction results and save the output reconstructed image
print("reconstruction results:")
print("the BO NMSE is:",nmse)
print("the BO SSIM is:",ssim)
print("the BO PSNR is:",psnr)

np.save(str(args.out_dir)+'/recon_img.npy',rcon)
np.save(str(args.out_dir)+'/recon_results.npy',np.array([nmse,ssim,psnr]))
plt.imshow(rcon[:,:],cmap='Greys_r')
plt.axis('off')
plt.savefig(str(args.out_dir)+'/recon_img.png')


