import torch
import numpy as np

import fastmri 
from fastmri.fftc import fft2c_new, ifft2c_new
from fastmri.math import complex_abs

from loss import mse,nmse,psnr,ssim

def zerofill_samp(img1_k,ind):
  #generate the zero-filled reconstructed images given the k-space sampled pixel index 
    #inputs: 
    #img_k1: original k-space data, size:(imgsz, imgsz)
    #ind: sampled pixel index 
    
  img_r = fastmri.complex_abs(ifft2c_new(img1_k)).view(-1,1)
  img_r = img_r.reshape(imgsz,imgsz)
  pixnum = len(ind)
  img1_k_ = img1_k[w:imgsz-w,w:imgsz-w,:]
  picked_r = img1_k_[:,:,0].reshape(-1,1)
  picked_i = img1_k_[:,:,1].reshape(-1,1)

  for i in range((imgsz-2*w)**2):
    if i not in ind:
      picked_r[i,0] = 0 
      picked_i[i,0] = 0 

  picked_r = picked_r.reshape(imgsz-2*w,imgsz-2*w)
  picked_i = picked_i.reshape(imgsz-2*w,imgsz-2*w)

  picked_k = torch.stack([picked_r,picked_i],dim = 2) 

  whole_k = torch.zeros(imgsz,imgsz,2)
  whole_k[w:imgsz-w,w:imgsz-w,:] = picked_k
  rcon = fastmri.complex_abs(ifft2c_new(whole_k))
  
  nmse_ =  nmse(img_r.numpy(),rcon.numpy())
  ssim_ =  ssim(img_r.numpy(),rcon.numpy())
  psnr_ = psnr(img_r.numpy(),rcon.numpy())
  

  return rcon,nmse_,ssim_,psnr_


def bayesian_stat_r_nor(imgz,w,img_k1,ind,sig,realcov_,imagcov_,realmean,imagmean,magmean,ind_previous = None):
  #generate the gpr inferred k-space mean and variance given the sampled pixel index 
    #inputs: 
    #img_k1: cropped k-space data, cropped size: (imgsz-2*w, imgsz-2*w), w: zero-padded width 
    #ind: sampled pixel index
    #sig: small value added to the covariane matrix diagonal part, default 1e-03
    #realcov_,imagcov_: covariance matrices of real and imaginary part of k-space
    #ind_previous: previous sampled pixel index 
  
  pixnum = len(ind)

  realcovmat = realcov_[ind,:][:,ind] + sig*torch.eye(pixnum)
  imagcovmat = imagcov_[ind,:][:,ind] + sig*torch.eye(pixnum)

  realcovvec = realcov_[:,ind]
  imagcovvec = imagcov_[:,ind]  

  for i in range(len(ind)):
    realcovmat[i,i] = realvar[ind[i]] + sig
    imagcovmat[i,i] = imagvar[ind[i]] + sig
    realcovvec[ind[i],i] = realvar[ind[i]]
    imagcovvec[ind[i],i] = imagvar[ind[i]]

  rmean = torch.matmul(realcovvec,torch.matmul(torch.inverse(realcovmat),(img_k1[:,:,0]/magmean-realmean).view(-1,1)[ind])) + realmean.view(-1,1)
  imean = torch.matmul(imagcovvec,torch.matmul(torch.inverse(imagcovmat),(img_k1[:,:,1]/magmean-imagmean).view(-1,1)[ind])) + imagmean.view(-1,1)

  rmean[ind] = ((img_k1[:,:,0]/magmean).reshape((imgsz-2*w)**2,1))[ind]
  imean[ind] = ((img_k1[:,:,1]/magmean).reshape((imgsz-2*w)**2,1))[ind]

  rvar = realvar - torch.sum(torch.matmul(realcov[:,ind],torch.inverse(realcovmat))*realcov[:,ind],1) + sig 
  ivar = imagvar - torch.sum(torch.matmul(imagcov[:,ind],torch.inverse(imagcovmat))*imagcov[:,ind],1) + sig

  if ind_previous != None: 
    ind = torch.cat([ind_previous,ind])

 
  kmean = torch.stack([(rmean).reshape(imgsz-2*w,imgsz-2*w)*magmean,\
                         (imean).reshape(imgsz-2*w,imgsz-2*w)*magmean],dim=2)   
  

  rmean = rmean.reshape(imgsz-2*w,imgsz-2*w)
  imean = imean.reshape(imgsz-2*w,imgsz-2*w)

  return rmean,imean,kmean,rvar,ivar

  

def bayesian_mean_samp_nor(imgsz,w,img1_k,ind,realcov,imagcov,realmean,imagmean,magmean,lamb):
  #generate the BO reconstructed images given the k-space sampled pixel index 

    #inputs: 
    #img_k1: original k-space data, size: (imgsz, imgsz)
    #ind: sampled pixel index
    #lamb: small value added to the covariane matrix diagonal part, default 1e-03
  
  pixnum = len(ind)

  img1_r = fastmri.complex_abs(ifft2c_new(img1_k)).view(-1,1)
  img1_r = img1_r.reshape(imgsz,imgsz)
  img1_k_ = (img1_k[w:imgsz-w,:,:])[:,w:imgsz-w,:]
  
  realcovmat = realcov[ind,:][:,ind] + lamb*torch.eye(pixnum)
  imagcovmat = imagcov[ind,:][:,ind] + lamb*torch.eye(pixnum)

  realcovvec = realcov[:,ind]
  imagcovvec = imagcov[:,ind]  
  rmean = torch.matmul(realcovvec,torch.matmul(torch.inverse(realcovmat),(img1_k_[:,:,0]/magmean-realmean).view(-1,1)[ind])) + realmean.reshape(-1,1)
  imean = torch.matmul(imagcovvec,torch.matmul(torch.inverse(imagcovmat),(img1_k_[:,:,1]/magmean-imagmean).view(-1,1)[ind])) + imagmean.reshape(-1,1)


  rmean[ind] = ((img1_k_[:,:,0]/magmean).reshape((imgsz-2*w)**2,1))[ind]
  imean[ind] = ((img1_k_[:,:,1]/magmean).reshape((imgsz-2*w)**2,1))[ind]
 
  kmean = torch.stack([(rmean).reshape(imgsz-2*w,imgsz-2*w)*magmean,\
                         (imean).reshape(imgsz-2*w,imgsz-2*w)*magmean],dim=2)   
  
  kmean_whole = torch.zeros(imgsz,imgsz,2)
  kmean_whole[w:imgsz-w,w:imgsz-w,:] = kmean
  bayesian_mean = fastmri.complex_abs(ifft2c_new(kmean_whole))
  
  nmse_ =  nmse(img1_r.numpy(),bayesian_mean.numpy())
  ssim_ =  ssim(img1_r.numpy(),bayesian_mean.numpy())
  psnr_ = psnr(img1_r.numpy(),bayesian_mean.numpy())
  
  return bayesian_mean,nmse_,ssim_,psnr_



def GPR_cir_nor(imgsz,w,img_k,round,thre,mode = None):
  #img_k: original size k-space image 
  img_r = fastmri.complex_abs(ifft2c_new(img_k)).view(-1,1)
  img_r = img_r.reshape(imgsz,imgsz)
  #crop the k-space image
  img_k_ = (img_k[w:imgsz-w,:,:])[:,w:imgsz-w,:]

  error_whole = torch.zeros(round,imgsz-2*w,imgsz-2*w) 
  errormag_list = torch.zeros(round,(imgsz - 2*w)//2+1)
  results_list = torch.zeros(round, 4)
  #img_list = torch.zeros(round,imgsz,imgsz)

  r_list = []

  realcov_ = torch.clone(realcov)
  imagcov_ = torch.clone(imagcov)
  realmean_ = torch.clone(realmean)
  imagmean_ = torch.clone(imagmean)

  realvar_ = torch.clone(realvar)
  imagvar_ = torch.clone(imagvar) 

  if mode == 'pix_var': 
    realvar_all = torch.zeros((imgsz-2*w)**2,round)
    imagvar_all = torch.zeros((imgsz-2*w)**2,round)
    magerr_all = torch.zeros((imgsz-2*w)**2,round)
  else: 
    realvar_all = 0
    imagvar_all = 0 
    magerr_all = 0 


  for i in range(round): 

    mag = torch.sqrt(realmean_**2 + imagmean_**2)
    if mode == 'pix_var': 
      realvar_all[:,i] = realvar_
      imagvar_all[:,i] = imagvar_

    realvar_[realvar_ < 0] = sig
    imagvar_[imagvar_ < 0] = sig
      
    errorreal = realvar_**0.5
    errorimag = imagvar_**0.5
        
    errormag = torch.sqrt((realmean_.view(-1))**2*errorreal**2+(imagmean_.view(-1))**2*errorimag**2)/(mag/magmean).view(-1)
    errormag[(mag*magmean).view(-1) == 0] = 0
    
    error_whole[i,:,:] = errormag.reshape(160,160)
    if mode == 'pix_var': 
      magerr_all[:,i] = errormag   

  #calculate the mean magnitude error and find the maximum position to sample 

    errormagmean = magerror_rrank(imgsz,w,errormag)
    errormag_list[i,:] = errormagmean

    sorted_error, ind_r = torch.sort(errormagmean, descending = True)
    for ind in ind_r: 
      if ind not in r_list: 
        r_ind = ind 
        break 
 
    ind0 = torch.from_numpy(r_dic[r_ind.item()])

    r_list.append(r_ind)
    #print('round:',i,',chosen r position:',r_ind.item())
    

    #update distribution 
    if i == 0: 
      
      realmean_,imagmean_,kmean,realvar_,imagvar_ = bayesian_stat_r_nor(img_k_,ind0,sig,realcov_,imagcov_,ind_previous = None)
      ind_list = ind0

    else: 
      ind_list = torch.cat([ind_list,ind0])
      if i != round - 1: 
          realmean_,imagmean_,kmean,realvar_,imagvar_  = bayesian_stat_r_nor(img_k_,ind_list,sig,realcov_,imagcov_,ind_previous = None)
 #kmean_:already denormolized, real&imagmean: normalized and divided by sqrt(real^2+imag^2)
      

    kmean_whole = torch.zeros(imgsz,imgsz,2)
    kmean_whole[w:imgsz-w,w:imgsz-w,:] = kmean


  return r_list, errormag_list, error_whole

