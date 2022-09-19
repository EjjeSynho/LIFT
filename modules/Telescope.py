# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:23:18 2020

@author: cheritie
"""
from matplotlib.colors import NoNorm
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import numexpr as ne
import inspect
import cupy as cp
from cupyx.scipy.fft import get_fft_plan
import cupyx.scipy.fft


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class Telescope:
    def __init__(self,
                 img_resolution,
                 pupil,
                 diameter,
                 focalLength,
                 pupilReflectivity = 1):
     
        self.img_resolution     = img_resolution  # Sampling of the telescope's PSF
        self.D                  = diameter        # Diameter in m
        self.f                  = focalLength     # Effective focal length of the telescope [m]
        self.object             = None

        self.pupil  = pupil
        assert self.pupil.shape[0] == self.pupil.shape[1], "Error: pupil mask must be a square array!"

        self.pupilReflectivity = pupilReflectivity    # A non uniform reflectivity can be input by the user
        self.src          = None                      # a source object associated to the telescope object
        self.det          = None
        self.tag          = 'telescope'               # a tag of the object
    
        self.area = np.pi * self.D**2 / 4
        self.fluxMap = lambda nPhotons, sampling_time: self.pupilReflectivity * self.pupil/np.sum(self.pupil) * nPhotons * self.area * sampling_time
        self.flux = lambda nPhotons, sampling_time: self.pupilReflectivity / np.sum(self.pupil) * nPhotons * self.area * sampling_time

        ident = 20
        char = '-'
        print(ident*char, 'TELESCOPE', ident*char)
        print('Diameter \t\t\t'+str(self.D) + ' \t [m]') 
        print('Pupil sampling \t\t\t'+str(self.pupil.shape[0]) + ' \t [pix]') 
        #print('PSF sampling \t\t\t'+str(self.img_resolution) + ' \t [pix]') 
        print(int(ident*2.4)*char)


    def PropagateFieldGPU(self, amplitude, phase, wavelength, return_intensity, oversampling=1):
        zeroPaddingFactor = self.f / self.det.pixel_size * wavelength / self.D
        resolution = self.pupil.shape[0]
            
        N = int(np.fix(zeroPaddingFactor * oversampling * resolution))

        if self.img_resolution * oversampling > N:
            print('Error: image sampling is too big for the pupil sampling')
            return None

        # If PSF is strongly undersampled, appply oversampling trick
        if zeroPaddingFactor * oversampling < 2:
            oversampling = 2.0 / zeroPaddingFactor

        if not hasattr(amplitude, 'device'):
            amplitude = cp.array(amplitude, dtype=cp.complex64)
        if not hasattr(phase, 'device'):
            phase = cp.array(phase, dtype=cp.complex64)
        
        pad_width = np.ceil((N-resolution)/2).astype('int')

        supportPadded = amplitude * cp.exp(1j*phase)
        supportPadded = cp.pad(supportPadded, pad_width=pad_width, constant_values=0)
        
        N = supportPadded.shape[0]
        img_size = np.ceil(self.img_resolution*oversampling).astype('int')

        # PSF computation
        [xx,yy] = cp.meshgrid( cp.linspace(0,N-1,N), cp.linspace(0,N-1,N), copy=False )    
        center_aligner = cp.exp(-1j*cp.pi/N * (xx+yy) * (1-self.img_resolution%2)).astype(cp.complex64)
        #                                                        ^--- this is to account odd/even number of pixels
        # Propagate with Fourier shifting
        PSF = cp.fft.fftshift(1/N * cp.fft.fft2(cp.fft.ifftshift(supportPadded*center_aligner)))

        if N%2 == img_size%2:
            shift_pix = 0
        else:
            if N%2 == 0:
                shift_pix = 1
            else:
                shift_pix = -1

        ids = np.array([np.ceil(N/2) - img_size//2 + (1-N%2)-1, np.ceil(N/2)+ img_size//2 + shift_pix]).astype('uint')
        PSF = cp.asnumpy( PSF[ids[0]:ids[1], ids[0]:ids[1]] ) # transfer data back to RAM from VRAM

        # Take oversampling into account
        if oversampling > 1:
            if return_intensity:
                PSF = np.abs(PSF)**2
                energy_before = np.sum( np.abs(PSF)**2 )
                PSF = resize(PSF, (self.img_resolution, self.img_resolution), anti_aliasing=True)
                energy_after = np.sum( np.abs(PSF)**2 )
                PSF *= (energy_before/energy_after) # to fix distribution of energy in pixels
            else:
                energy_before = np.sum( np.abs(PSF)**2 )
                Re = resize(np.real(PSF), (self.img_resolution, self.img_resolution), anti_aliasing=True)
                Im = resize(np.imag(PSF), (self.img_resolution, self.img_resolution), anti_aliasing=True)
                PSF = Re + 1j*Im
                energy_after = np.sum( np.abs(PSF)**2 )
                PSF *= (energy_before/energy_after) # to fix distribution of energy in pixels
        else:
            if return_intensity:
                PSF = np.abs(PSF)**2
        return PSF


    def PropagateField(self, amplitude, phase, wavelength, return_intensity, oversampling=1):

        zeroPaddingFactor = self.f / self.det.pixel_size * wavelength / self.D
        resolution = self.pupil.shape[0]
    
        if self.img_resolution > zeroPaddingFactor*resolution:
            print('Error: image sampling is too big for the pupil sampling. Try using more pixels in pupil mask')
            return None

        # If PSF is strongly undersampled, appply the oversampling trick
        if zeroPaddingFactor * oversampling < 2:
            oversampling = 2.0 / zeroPaddingFactor

        N = np.fix(zeroPaddingFactor * oversampling * resolution).astype('int')
        pad_width = np.ceil((N-resolution)/2).astype('int')
        supportPadded = np.pad(amplitude * np.exp(1j*phase), pad_width=((pad_width,pad_width),(pad_width,pad_width)), constant_values=0)
        N = supportPadded.shape[0] # make sure the number of pxels is correct after the padding
        img_size = np.ceil(self.img_resolution*oversampling).astype('int')

        # PSF computation
        [xx,yy] = np.meshgrid( np.linspace(0,N-1,N), np.linspace(0,N-1,N) )        
        center_aligner = np.exp(-1j*np.pi/N * (xx+yy) * (1-self.img_resolution%2))
        #                                                        ^--- this is to account odd/even number of pixels

        # Propagate with shifting the Fourier spectrum
        PSF = np.fft.fftshift(1/N * np.fft.fft2(np.fft.ifftshift(supportPadded*center_aligner)))

        if N%2 == img_size%2:
            shift_pix = 0
        else:
            if N%2 == 0:
                shift_pix = 1
            else:
                shift_pix = -1

        ids = np.array([np.ceil(N/2) - img_size//2 + (1-N%2)-1, np.ceil(N/2)+ img_size//2 + shift_pix]).astype('uint')
        PSF = PSF[ids[0]:ids[1], ids[0]:ids[1]]

        # Take oversampling into the account
        if oversampling > 1:
            if return_intensity:
                PSF = np.abs(PSF)**2
                energy_before = np.sum( np.abs(PSF)**2 )
                PSF = resize(PSF, (self.img_resolution, self.img_resolution), anti_aliasing=True)
                energy_after = np.sum( np.abs(PSF)**2 )
                PSF *= (energy_before/energy_after) # to fix distribution of energy in pixels
            else:
                energy_before = np.sum( np.abs(PSF)**2 )
                Re = resize(np.real(PSF), (self.img_resolution, self.img_resolution), anti_aliasing=True)
                Im = resize(np.imag(PSF), (self.img_resolution, self.img_resolution), anti_aliasing=True)
                PSF = Re + 1j*Im
                energy_after = np.sum( np.abs(PSF)**2 )
                PSF *= (energy_before/energy_after) # to fix distribution of energy in pixels
        else:
            if return_intensity:
                PSF = np.abs(PSF)**2
        return PSF


    def ComputePSF(self, intensity=True, oversampling=1, GPU=False, polychrome=False):
        if self.src.tag != 'source':
            print('Error: no proper source object is attached')
            return None

        if GPU:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

        PSF_chromatic = []
        for point in self.src.spectrum:
            phase = 2 * np.pi / point['wavelength'] * self.src.OPD
            amplitude = np.sqrt(self.flux(point['flux'], self.det.sampling_time)) * self.pupil
            #if len(phase.shape) == 3: # make sure it's not [N x N x 1]
            #    phase = np.squeeze(phase, axis=2)
            if GPU:
                PSF_chromatic.append( self.PropagateFieldGPU(amplitude, phase, point['wavelength'], intensity, oversampling) )
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks() # clear GPU memory
            else:
                PSF_chromatic.append( self.PropagateField(amplitude, phase, point['wavelength'], intensity, oversampling) )

        PSF_chromatic = np.dstack(PSF_chromatic)

        if intensity:
            if polychrome:
                return PSF_chromatic
            else:
                return PSF_chromatic.sum(axis=2)
        else:
            if PSF_chromatic.shape[2] == 1:
                return PSF_chromatic[:,:,0]
            else:
                return PSF_chromatic


    def PropagateFieldBatchGPU(self, OPD, wavelength, amplitude=1.):
        oversampling = 1 # Yet unsupported
        phase = cp.array( OPD, dtype=cp.float32 ) * 2*np.pi/wavelength

        zeroPaddingFactor = self.f / self.det.pixel_size * wavelength / self.D
        resolution = self.pupil.shape[0]

        N = int(np.fix(zeroPaddingFactor * oversampling * resolution))

        if self.img_resolution * oversampling > N:
            print('Error: image sampling is too big for the pupil sampling')
            return None

        # If PSF is strongly undersampled, appply oversampling trick
        if zeroPaddingFactor * oversampling < 2:
            oversampling = 2.0 / zeroPaddingFactor

        pad_width = np.ceil((N-resolution)/2).astype('int')

        supportPadded = cp.zeros([resolution+2*pad_width, resolution+2*pad_width, phase.shape[2]], dtype=cp.complex64)
        supportPadded[pad_width:pad_width+resolution, pad_width:pad_width+resolution,:] = amplitude * cp.exp(1j*phase)

        #del amplitude, phase

        N = supportPadded.shape[0]
        img_size = np.ceil(self.img_resolution*oversampling).astype('int')

        # PSF computation
        [xx,yy] = cp.meshgrid( cp.linspace(0,N-1,N), cp.linspace(0,N-1,N), copy=False)
        center_aligner = cp.expand_dims(cp.exp(-1j*cp.pi/N * (xx+yy) * (1-self.img_resolution%2)), axis=2).astype(cp.complex64)
        #                                                        ^--- this is to account odd/even number of pixels

        # Propagate with Fourier shifting
        plan = get_fft_plan(supportPadded, axes=(0, 1), value_type='C2C')  # for batched, C2C, 2D transform
        supportPadded = cupyx.scipy.fft.ifftshift(supportPadded * center_aligner)
        PSF = cupyx.scipy.fft.fftshift(1/N * cupyx.scipy.fft.fft2(supportPadded, axes=(0, 1), plan=plan))

        #del supportPadded, center_aligner, xx, yy

        if N%2 == img_size%2:
            shift_pix = 0
        else:
            if N%2 == 0: shift_pix = 1
            else: shift_pix = -1

        ids = np.array([np.ceil(N/2) - img_size//2 + (1-N%2)-1, np.ceil(N/2)+ img_size//2 + shift_pix]).astype('uint')
        PSF = PSF[ids[0]:ids[1], ids[0]:ids[1]]

        # Take oversampling into account
        if oversampling > 1:
            raise NotImplementedError('Error: unsupported feature')

        PSF = cp.sum(cp.abs(PSF)**2, axis=2)
        return PSF


    def ComputePSFBatchGPU(self, OPD=None, polychrome=False):
        if self.src.tag != 'source':
            print('Error: no proper source object is attached')
            return None

        if OPD is None:
            OPD = self.src.OPD

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        PSF_spectral = None
        amplitude = cp.atleast_3d( cp.array(self.pupil, dtype=cp.float32) ) # [N x N x 1]
        try:
            PSF_spectral = []
            for point in self.src.spectrum:
                wavelength = point['wavelength']
                flux = self.flux(point['flux'], self.det.sampling_time)
                PSF_spectral.append( self.PropagateFieldBatchGPU(OPD, wavelength,  np.sqrt(flux)*amplitude) )

                #mempool.free_all_blocks()
                #pinned_mempool.free_all_blocks() # clear GPU memory

            if polychrome:
                PSF_spectral = cp.asnumpy( cp.dstack(PSF_spectral) )
            else:
                PSF_spectral = cp.asnumpy( cp.sum(cp.dstack(PSF_spectral), axis=2) )

        except cp.cuda.memory.OutOfMemoryError:
            print('Oops, we run out of VRAM')
        
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return PSF_spectral


    def GetAxesPSF(self, PSF, units):
        angular_size = self.det.pixel_size / self.f
        fl = float(not PSF.shape[0]%2)

        limits_x = np.array( [-PSF.shape[0]/2, PSF.shape[0]/2-fl]) + 0.5*fl
        limits_y = np.array( [-PSF.shape[1]/2, PSF.shape[1]/2-fl]) + 0.5*fl
        extent   = np.array( [limits_x[0], limits_x[1], limits_y[0], limits_y[1]] )

        if units == 'microns' or units == 'micrometers' or units == 'micron' or units == 'micrometer':
            extent *= self.det.pixel_size
        elif units == 'arcsec' or units == 'arcseconds' or units == 'asec' or units == 'arcsecond':
            extent *= angular_size * 206264.8
        elif units == 'mas' or units == 'milliarcseconds' or units == 'milliarcsecond':
            extent *= angular_size * 206264806.71915
        elif units == 'pixels' or units == 'pix' or units == 'pixel':
            extent =  extent
        else:
            print('Warning: unrecognized unit "'+units+'". Displaying in pixels...')

        return extent


    def resetOPD(self):
        # re-initialize the telescope OPD to a flat wavefront
        self.OPD = np.copy(self.pupil)
    

