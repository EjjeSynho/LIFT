# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:33:20 2020

@author: cheritie
"""

import numpy as np
import os 
import skimage.transform as sk

def createFolder(path):
    
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed:" % path)
        if path:
            print('Directory already exists!')
        else:
            print('Maybe you do not have access to this location.')
    else:
        print ("Successfully created the directory %s !" % path)

def emptyClass():
    class nameClass:
        pass
    return nameClass

def bsxfunMinus(a,b):      
    A =np.tile(a[...,None],len(b))
    B =np.tile(b[...,None],len(a))
    out = A-B.T
    return out



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def translationImageMatrix(image,shift):
    # translate the image with the corresponding shift value
    tf_shift = sk.SimilarityTransform(translation=shift)    
    return tf_shift

def globalTransformation(image,shiftMatrix,order=3):
        output  = sk.warp(image,(shiftMatrix).inverse,order=order)
        return output


def reshape_2D(A,axis = 2, pupil=False ):
    if axis ==2:
        out = np.reshape(A,[A.shape[0]*A.shape[1],A.shape[2]])
    else:
        out = np.reshape(A,[A.shape[0],A.shape[1]*A.shape[2]])
    if pupil:
        out = np.squeeze(out[pupil,:])
    return out


def reshape_3D(A,axis = 1 ):
    if axis ==1:
        dim_rep =np.sqrt(A.shape[0]) 
        out = np.reshape(A,[dim_rep,dim_rep,A.shape[1]])
    else:
        dim_rep =np.sqrt(A.shape[1]) 
        out = np.reshape(A,[A.shape[0],dim_rep,dim_rep])    
    return out        
    
    
    