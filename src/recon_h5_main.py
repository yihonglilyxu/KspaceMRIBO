import argparse
import pathlib

import torch
from torch.utils.data import Dataset, DataLoader
import h5py

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
from h5_dataloader import MRIDataset_h5

import time

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--imgsz",default = 256,help = "image size")
    parser.add_argument("--w",default = 48)
    
    parser.add_argument("--imgpath",type=str,help="The path of k-space MRI images h5 dataset", required=True)
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

#load k-space MRI images h5 dataset. Size: (n,imgsz,imgsz)
dataset = MRIDataset_h5(in_file = args.imgpath,mode = 'real_imag')
n = len(dataset)
print('total testing images #:',n)

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {} 

bs = 1
train_loader = DataLoader(dataset, batch_size=bs, drop_last=False, shuffle= False,**kwargs)

#load the undersampling pattern
rpattern = np.load(args.undersampled_path)
r_dic = rdic(imgsz,w)
mask_r,mask_r_ratio= mask_of_r_order(imgsz-2*w,rpattern,r_dic)

lamb = 1e-3
ind = np.linspace(0,(imgsz-2*w)**2-1,(imgsz-2*w)**2,dtype = int)

#start reconstruction
r_nor_results = torch.zeros(n,3)
t1 = time.time()

imgsavepath = str(args.out_dir)+ "/reconstructed_images/"
isExist = os.path.exists(imgsavepath)
if not isExist:

   os.makedirs(imgsavepath)
   print("The reconstructed image folder is created!")


for j,image in enumerate(train_loader):
    img_k = image[0,:,:,:]
    rcon,nmse,ssim,psnr = bayesian_mean_samp_nor(imgsz,w,img_k,ind[mask_r.view(-1) == 1],realcov,imagcov,realmean,imagmean,magmean,lamb)
    r_nor_results[j,:] = torch.tensor([nmse,ssim,psnr])
    plt.imshow(rcon[:,:],cmap='Greys_r')
    plt.axis('off')
    plt.savefig(imgsavepath+str(j)+'.png')
    print('image',str(j),'reconstruction finished')
    print('reconstruction results (NMSE,SSIM,PSNR) are',nmse.item(),ssim.item(),psnr.item())
    print('--*50')

t2 = time.time()
print('reconstruction finished')
print('total reconstruction time:', t2-t1,'s')
print('average reconstruction results are',)
print('mean nmse:',torch.mean(r_nor_results[:,0]).item(),'mean ssim:',torch.mean(r_nor_results[:,1]).item(), \
                                                'mean psnr:',torch.mean(r_nor_results[:,2]).item())

np.save(str(args.out_dir)+'/recon_results.npy',r_nor_results)



