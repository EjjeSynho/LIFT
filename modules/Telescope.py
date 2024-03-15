import sys
sys.path.insert(0, '..')

import numpy as np

import sys
try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.fftpack import get_fft_plan
    global_gpu_flag = True
    
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np
    global_gpu_flag = False

from tools.misc import binning


class Telescope:
    def __init__(self,
                 img_resolution,
                 pupil,
                 diameter,
                 focalLength,
                 pupilReflectivity = 1.0,
                 gpu_flag = False):

        global global_gpu_flag
        self.pupil = pupil
        assert self.pupil.shape[0] == self.pupil.shape[1], "Error: pupil mask must be a square array!"

        self.img_resolution    = img_resolution    # Sampling of the telescope's PSF
        self.pupilReflectivity = pupilReflectivity # A non uniform reflectivity can be input by the user
        self.D                 = diameter          # Diameter in m
        self.f                 = focalLength       # Effective focal length of the telescope [m]
        self.object            = None              # 2-d matrix to be convolved with the PSF 
        self.src               = None              # A Source object attached to the telescope object
        self.det               = None              # A Detector object attached to the telescope object
        self.tag               = 'telescope'       # A tag of the object
        self.oversampling      = 1                 # minimal PSF oversampling 
        self.gpu               = gpu_flag and global_gpu_flag

        # if self.gpu:
        #     self.pupil = cp.array(self.pupil, dtype=cp.float32)
        #     if self.object is not None:
        #         self.object = cp.array(self.tel.object, cp.float32)

        self.area = np.pi * self.D**2 / 4
        self.fluxMap = lambda nPhotons, sampling_time: self.pupilReflectivity * self.pupil/self.pupil.sum() * nPhotons * self.area * sampling_time
        self.flux = lambda nPhotons, sampling_time: self.pupilReflectivity /self.pupil.sum() * nPhotons * self.area * sampling_time

        ident = 20
        char = '-'
        print(ident*char, 'TELESCOPE', ident*char)
        print('Diameter \t\t\t'+str(self.D) + ' \t [m]') 
        print('Pupil sampling \t\t\t'+str(self.pupil.shape[0]) + ' \t [pix]') 
        print(int(ident*2.4)*char)


    @property
    def gpu(self):
        return self.__gpu
    

    @gpu.setter
    def gpu(self, var):
        if var:
            self.__gpu = True
            if hasattr(self, 'pupil'):
                if not hasattr(self.pupil, 'device'):
                    self.pupil = cp.array(self.pupil, dtype=cp.float32)
               
            if self.object is not None:
                if not hasattr(self.object, 'device'):
                    self.object = cp.array(self.object, dtype=cp.float32)
            
        else:
            self.__gpu = False
            if hasattr(self, 'pupil'):
                if hasattr(self.pupil, 'device'):
                    self.pupil = self.pupil.get()

            if self.object is not None:
                if hasattr(self.object, 'device'):
                    self.object = self.object.get()


    def PropagateField(self, amplitude, phase, wavelength, return_intensity, oversampling=None):
        xp = cp if self.gpu else np

        zeroPaddingFactor = self.f / self.det.pixel_size * wavelength / self.D
        resolution = self.pupil.shape[0]
        if oversampling is not None: self.oversampling = oversampling

        if self.img_resolution > zeroPaddingFactor*resolution:
            print('Error: image has too many pixels for this pupil sampling. Try using a pupil mask with more pixels')
            return None

        # If PSF is undersampled apply the integer oversampling
        if zeroPaddingFactor * self.oversampling < 2:
            self.oversampling = (np.ceil(2.0/zeroPaddingFactor)).astype('int')
        
        # This is to ensure that PSF will be binned properly if number of pixels is odd
        if self.oversampling % 2 != self.img_resolution % 2:
            self.oversampling += 1

        img_size = np.ceil(self.img_resolution*self.oversampling).astype('int')
        N = np.fix(zeroPaddingFactor * self.oversampling * resolution).astype('int')
        pad_width = np.ceil((N-resolution)/2).astype('int')

        if not hasattr(amplitude, 'device'): amplitude = xp.array(amplitude, dtype=cp.float32)
        if not hasattr(phase, 'device'):     phase     = xp.array(phase, dtype=cp.complex64)
        
        #supportPadded = cp.pad(amplitude * cp.exp(1j*phase), pad_width=pad_width, constant_values=0)
        supportPadded = xp.pad(amplitude * xp.exp(1j*phase), pad_width=((pad_width,pad_width),(pad_width,pad_width)), constant_values=0)
        N = supportPadded.shape[0] # make sure the number of pxels is correct after the padding

        # PSF computation
        [xx,yy] = xp.meshgrid( xp.linspace(0,N-1,N), xp.linspace(0,N-1,N), copy=False )    
        center_aligner = xp.exp(-1j*xp.pi/N * (xx+yy) * (1-self.img_resolution%2)).astype(xp.complex64)
        #                                                        ^--- this is to account odd/even number of pixels
        # Propagate with Fourier shifting
        EMF = xp.fft.fftshift(1/N * xp.fft.fft2(xp.fft.ifftshift(supportPadded*center_aligner)))

        # Again, this is to properly crop a PSF with the odd/even number of pixels
        if N % 2 == img_size % 2:
            shift_pix = 0
        else:
            if N % 2 == 0: shift_pix = 1
            else: shift_pix = -1

        # Support only rectangular PSFs
        ids = xp.array([np.ceil(N/2) - img_size//2+(1-N%2)-1, np.ceil(N/2) + img_size//2+shift_pix]).astype(xp.int32)
        EMF = EMF[ids[0]:ids[1], ids[0]:ids[1]]

        if return_intensity:
            return binning(xp.abs(EMF)**2, self.oversampling)

        return EMF # in this case, raw electromagnetic field is returned. It can't be simpli binned


    def ComputePSF(self, intensity=True, polychrome=True, oversampling=1):
        xp = cp if self.gpu else np

        if self.src.tag != 'source':
            print('Error: no proper source object is attached')
            return None

        if self.gpu:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

        PSF_chromatic = []
        for point in self.src.spectrum:
            phase = 2 * xp.pi / point['wavelength'] * self.src.OPD
            amplitude = xp.sqrt(self.flux(point['flux'], self.det.sampling_time)) * self.pupil
            PSF_chromatic.append( self.PropagateField(amplitude, phase, point['wavelength'], intensity, oversampling) )
            
            if self.gpu: # clear GPU buffers
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks() # clear GPU memory

        PSF_chromatic = xp.dstack(PSF_chromatic)

        if intensity:
            if polychrome:
                return PSF_chromatic.sum(axis=2)
            else:
                return PSF_chromatic
        else:
            if PSF_chromatic.shape[2] == 1:
                return PSF_chromatic.squeeze(2)
            else:
                return PSF_chromatic.squeeze(2)


    def PropagateFieldBatch(self, OPD, wavelength, amplitude=1., return_batch=True):
        xp = cp if self.gpu else np
        oversampling = 1 # Yet unsupported
        
        if hasattr(OPD, 'device'):
            phase = OPD* 2*xp.pi/wavelength
        else:
            phase = xp.array( OPD, dtype=xp.float32 ) * 2*xp.pi/wavelength

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

        supportPadded = xp.zeros([resolution+2*pad_width, resolution+2*pad_width, phase.shape[2]], dtype=xp.complex64)
        supportPadded[pad_width:pad_width+resolution, pad_width:pad_width+resolution,:] = amplitude * xp.exp(1j*phase)

        N = supportPadded.shape[0]
        img_size = np.ceil(self.img_resolution*oversampling).astype('int')

        # PSF computation
        [xx,yy] = xp.meshgrid( xp.linspace(0,N-1,N), xp.linspace(0,N-1,N), copy=False)
        center_aligner = xp.expand_dims(xp.exp(-1j*xp.pi/N * (xx+yy) * (1-self.img_resolution%2)), axis=2).astype(xp.complex64)
        #                                                        ^--- this is to account odd/even number of pixels

        # Propagate with Fourier shifting
        if self.gpu:
            plan = get_fft_plan(supportPadded, axes=(0, 1), value_type='C2C')  # for batched, C2C, 2D transform
            supportPadded = cupyx.scipy.fft.ifftshift(supportPadded * center_aligner)
            PSF = cupyx.scipy.fft.fftshift(1/N * cupyx.scipy.fft.fft2(supportPadded, axes=(0, 1), plan=plan))
        else:
            supportPadded = np.fft.ifftshift(supportPadded * center_aligner)
            PSF = np.fft.fftshift(1/N * np.fft.fft2(supportPadded, axes=(0, 1), plan=plan))

        #del supportPadded, center_aligner, xx, yy

        if N % 2 == img_size%2:
            shift_pix = 0
        else:
            if N%2 == 0: shift_pix = 1
            else: shift_pix = -1

        ids = np.array([np.ceil(N/2) - img_size//2 + (1-N%2)-1, np.ceil(N/2)+ img_size//2 + shift_pix]).astype('uint')
        PSF = PSF[ids[0]:ids[1], ids[0]:ids[1]]

        # Take oversampling into account
        if oversampling > 1:
            raise NotImplementedError('Error: unsupported feature')

        if return_batch:
            return xp.abs(PSF)**2
        else:
            return xp.sum(xp.abs(PSF)**2, axis=2)


    def ComputePSFBatch(self, OPD=None, polychrome=True, return_batch=False, return_GPU=False):
        xp = cp if self.gpu else np
        
        if self.src.tag != 'source':
            print('Error: no proper source object is attached')
            return None

        if OPD is None:
            OPD = self.src.OPD

        if self.gpu:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
        PSF_spectral = None
        
        amplitude = xp.atleast_3d(self.pupil)
        
        if self.gpu:
            amplitude = cp.array(amplitude, dtype=cp.float32) # [N x N x 1]

        try:
            PSF_spectral = []
            for point in self.src.spectrum:
                wavelength = point['wavelength']
                flux = self.flux(point['flux'], self.det.sampling_time)
                PSF_spectral.append( self.PropagateFieldBatch(OPD, wavelength,  np.sqrt(flux)*amplitude, return_batch) )

            return_format = lambda x: x if return_GPU else cp.asnumpy(x)

            if return_batch:
                PSF_spectral = return_format( xp.stack(PSF_spectral).transpose([1,2,3,0]) )
            else:
                PSF_spectral = return_format( xp.dstack(PSF_spectral) )

            if polychrome:
                PSF_spectral = PSF_spectral.sum(axis=-1)

        except cp.cuda.memory.OutOfMemoryError:
            print('Not enough VRAM')
        
        if self.gpu:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
        return PSF_spectral


    def GetAxesPSF(self, PSF, units):
        angular_size = self.det.pixel_size / self.f
        fl = float(not PSF.shape[0]%2) # flag to check if PSF has even number of pixels

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
        self.OPD = self.pupil.copy() # re-initialize the telescope OPD to a flat wavefront