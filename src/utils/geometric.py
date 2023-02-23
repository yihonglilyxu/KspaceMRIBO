import torch
import numpy as np

#some useful geometric function
#ind: pixel index
#x,y: 2d position in k-space
#phi: radial position phi in k-space

def ind_to_ij(ind): 
  return ind//(imgsz-2*w),ind%(imgsz-2*w)

def ind_to_xy(ind): 
  i = (ind//(imgsz-2*w))
  j = (ind%(imgsz-2*w))
  y = (imgsz-2*w-1-i) - (imgsz-2*w - 1)/2
  x = j - (imgsz-2*w - 1)/2
  return x,y

def ij_to_phi(i,j): 
  i_ = (imgsz-2*w-1-i) - (imgsz-2*w - 1)/2
  j_ = j - (imgsz-2*w - 1)/2
  #phi: size(1,#pixels)
  return torch.atan2(i_,j_)

def xy_to_ind(x,y): 
  i = (imgsz-2*w-1) - (imgsz-2*w - 1)/2 - y 
  j = x + (imgsz-2*w - 1)/2
  ind = i*(imgsz-2*w) + j
  return ind

def true_her(x,y):
 #find the Hemitian point in k-space given the target point
    return 1-x, -1-y
