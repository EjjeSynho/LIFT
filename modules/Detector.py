# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:37:29 2020

@author: cheritie
"""

import numpy as np
from numpy.core.fromnumeric import shape
from numpy.random import Generator, PCG64
import inspect
#from tools.real_data import GetDetectorMaps
#from tools.fit_gaussian import FitAndPlotGauss1D
import matplotlib.pyplot as plt
from scipy import signal as sg
import cupy as cp
from cupyx.scipy.signal import convolve2d

class Detector:
    def __init__(self, pixel_size, sampling_time, samples=1, RON=0, QE=1):
        self.QE            = QE #TODO: use QE?
        self.pixel_size    = pixel_size
        self.readoutNoise  = RON
        self.sampling_time = sampling_time
        self.samples       = samples   
        self.tag           = 'detector'        
        self.object        = None

        self.GPU           = False
        self.object_GPU    = None
        self.PSF_GPU       = None

    def getFrame(self, PSF, noise=True, integrate=True):
        if self.object is not None:
            if not self.GPU:
                PSF = sg.convolve2d(PSF, self.object, boundary='symm', mode='same') / self.object.sum()
            else:
                self.PSF_GPU = cp.array(PSF, dtype=cp.float32)
                self.object_GPU = cp.array(self.object, dtype=cp.float32)
                self.PSF_GPU = convolve2d(self.PSF_GPU, self.object_GPU, mode='same', boundary='symm') / self.object_GPU.sum() #, fillvalue=0)
                PSF = cp.asnumpy(self.PSF_GPU)

        if noise: 
            R_n = PSF + self.readoutNoise
            rng = np.random.default_rng()
            rg  = Generator(PCG64())
        else:
            R_n = np.ones(PSF.shape)
        
        # Compose the image cube
        image_cube = []
        for i in range(self.samples):
            if noise:
                photon = rng.poisson(PSF, PSF.shape)
                ron = rg.standard_normal(PSF.shape) * np.sqrt(self.readoutNoise)
                image_cube.append(photon + ron)
            else:
                image_cube.append(PSF)

        image_cube = np.dstack(image_cube)

        if integrate:
            image_cube = image_cube.mean(axis=2)
            if noise:
                R_n /= self.samples 

        return image_cube, R_n


    def __mul__(self, tel):
        tel.det = self
        self.object = tel.object

        return tel
