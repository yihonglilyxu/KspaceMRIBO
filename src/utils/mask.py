import torch
import numpy as np

######### Geometric Function #########
def mask_of_r(imgsz,r):
  mask = torch.zeros(imgsz,imgsz)

  for i in range(imgsz):
    for j in range(imgsz): 
      i_ = i - (imgsz-1)/2
      j_ = j - (imgsz-1)/2
    
      if (i_**2+j_**2)**0.5 >= r and (i_**2+j_**2)**0.5 < r+1: 
  #  if np.round((i_**2+j_**2)**0.5) == r:
        mask[i,j] = 1 

  return mask

def mask_in_r(imgsz,r):
  #generate the undersampling mask with solid circle sampled area of radii r
    #inputs:
    #imgsz: image size
    #r: radii 
  mask = torch.zeros(imgsz,imgsz)

  for i in range(imgsz):
    for j in range(imgsz): 
      i_ = i - (imgsz-1)/2
      j_ = j - (imgsz-1)/2
    
      if (i_**2+j_**2)**0.5 <= r:  
        mask[i,j] = 1 
  return mask 

def mask_of_phi(imgsz,r_low,r_high,phi_low,phi_up):
  #generate the undersampling mask with solid rings sampled area within r_low<r<=r_high and phi_low<=phi<phi_up
    #inputs:
    #imgsz: image size
    #r_low,r_high: lower and upper bound of radii
    #phi_low, phi_up: lower and upper bound of radial phi position 
    
  mask = torch.zeros(imgsz,imgsz)
  for i in range(imgsz):
    for j in range(imgsz): 
      i_ = torch.tensor([(imgsz-1-i) - (imgsz - 1)/2])
      j_ = torch.tensor([j - (imgsz - 1)/2])
      if torch.sqrt(i_**2+j_**2) > r_low and torch.sqrt(i_**2+j_**2) <= r_high: 
        if phi_low <= (torch.atan(i_/j_)) and (torch.atan(i_/j_)) < phi_up: 
          mask[i,j] = 1 

  return mask 

def rdic(imgsz,w):
  ##make r index dictionary##make r index dictionary
    r_dic = {}
    rlist = np.linspace(0,int((imgsz-2*w)/2),int((imgsz-2*w)/2)+1,dtype=int)
    indlist = np.linspace(0,(imgsz-2*w)**2-1,(imgsz-2*w)**2,dtype=int)
    for i in rlist: 
        mask = mask_of_r(imgsz-2*w,i)
        r_dic[i] = indlist[mask.view(-1) == 1]
    return r_dic


def mask_of_r_order(imgsz,rlist,r_dic):
  #generate the undersampling mask with circular sampled area given a list of sampling rings' radii 
    #inputs:
    #imgsz: cropped image size, size: (imgsz - 2*w, imgsz - 2*w)
    #rlist: a list of sampling rings' radii  
  mask = torch.zeros(imgsz,imgsz).view(-1)
  ratio = torch.zeros(len(rlist))
  n = 0
  for i in rlist: 
    mask[r_dic[int(i)]] = 1
    ratio[n] = torch.sum(mask == 1)/256**2
    n += 1

  mask = mask.reshape(imgsz,imgsz)

  return mask, ratio


def mask_larger(imgsz,w,mask):
 #generate the undersampling mask in the origianl image size (imgsz,imgsz)
    #inputs:
    #imgsz: image size
    #w: k-space image cropped width
    #mask: undersampling mask in cropped k-space, size: (imgsz - 2*w, imgsz -2*w)
  mask_ = torch.zeros(imgsz,imgsz)
  mask_[w:imgsz-w,w:imgsz-w] = mask
  return mask_



