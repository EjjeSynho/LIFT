#%%
import pickle
import numpy as np
import os
from tools.fit_gaussian import gaussian


def GetKLphaseScreens(number, id=0, path=None, sample=None):
    
    if path is None:
        path = 'C:/Users/akuznets/Data/AOF/Modes/Data_AOF_Johannes (Mar 2021)/Data4JB/'

    if sample is None:
        sample = 'NFM_DSM_Positions_121019_Ki0p4_Seeing0p57'
    
    path = os.path.normpath(path)
    path_KL_modes = os.path.join(path, 'KLmodes.pickle')
    sequence_path = os.path.join(path, sample + '.pickle')

    with open(path_KL_modes, 'rb') as handle:
        KL_modes, _, _ = pickle.load(handle)

    with open(sequence_path, 'rb') as handle:
        _, sequence_KL = pickle.load(handle)

    phases = KL_modes @ (sequence_KL[:,id+1:id+number+1]-sequence_KL[:,id:id+number])
    return phases.reshape([240,240,number])


def GenerateObject(sigma_obj=None):
    obj_resolution = 16
    xx, yy = np.meshgrid(np.arange(0, obj_resolution), np.arange(0, obj_resolution))
    if sigma_obj is None:
        return None    
    elif hasattr(sigma_obj, '__len__'):
        obj1 = np.abs( gaussian(1.0, obj_resolution/2-1, obj_resolution/2-1,  sigma_obj[0], sigma_obj[1])(xx,yy) )
    else:
        obj1 = np.abs( gaussian(1.0, obj_resolution/2-1, obj_resolution/2-1,  sigma_obj, sigma_obj)(xx,yy) )
    return obj1[:-1,:-1]


def ObjectTilted(amp=1.0, x_0=0.0, y_0=0.0, s_x=1.0, s_y=1.0, ang=0.0):
    #FWHM_x=2*np.sqrt(2*np.log(2)), FWHM_y=2*np.sqrt(2*np.log(2)), ang=0.):
    #s_x = FWHM_x / (2*np.sqrt(2*np.log(2)))
    #s_y = FWHM_y / (2*np.sqrt(2*np.log(2)))

    if s_x < 1e-3 or s_y < 1e-3: return None

    obj_resolution = 15
    lin_space = np.arange(-obj_resolution//2+1, obj_resolution//2+1)
    xx, yy = np.meshgrid(lin_space, lin_space)

    ang1 = ang*np.pi/180.0
    A =  np.cos(ang1)**2 / (2*s_x**2) + np.sin(ang1)**2 / (2*s_y**2)
    B = -np.sin(2*ang1)  / (4*s_x**2) + np.sin(2*ang1)  / (4*s_y**2)
    C =  np.sin(ang1)**2 / (2*s_x**2) + np.cos(ang1)**2 / (2*s_y**2)
    return amp * np.exp(-(A*(xx-x_0)**2 + 2*B*(xx-x_0)*(yy-y_0) + C*(yy-y_0)**2))


def magnitudeFromPSF(tel, photons, band, sampling_time=None):
    if sampling_time is None:
        sampling_time = tel.det.sampling_time
    zero_point = tel.src.PhotometricParameters(band)[2]
    fluxMap = photons / tel.pupil.sum() * tel.pupil
    nPhoton = np.nansum(fluxMap / tel.pupilReflectivity) / (np.pi*(tel.D/2)**2) / sampling_time
    return -2.5 * np.log10(368 * nPhoton / zero_point )


def TruePhotonsFromMag(tel, mag, band, sampling_time): # [photons/aperture] !not per m2!
    c = tel.pupilReflectivity*np.pi*(tel.D/2)**2*sampling_time
    return tel.src.PhotometricParameters(band)[2]/368 * 10**(-mag/2.5) * c


def PSF_from_KL(tel, residual_phase, diversity_phase, Z, NCPA_OPDs, exposure_time, residuals, randres=True):
    computed_samples = 100
    id = np.abs(np.random.randint(residual_phase.shape[2])-computed_samples-1).astype('uint') * int(randres)
    exposure_per_computed_sample = exposure_time / computed_samples # [s]
    buf_time = tel.det.sampling_time
    tel.det.sampling_time = exposure_per_computed_sample
    NCPAs = np.sum(Z.modesFullRes*NCPA_OPDs, axis=2, keepdims=True) # [m]
    tel.src.OPD = (residual_phase[:,:,id:computed_samples+id]*int(residuals)*0.5 + np.atleast_3d(diversity_phase) + NCPAs)
    PSF = tel.ComputePSFBatchGPU()
    tel.det.sampling_time = buf_time
    return PSF, id


def NoisyPSF(tel, PSF, integrate=True):
    return tel.det.getFrame(PSF, noise=True, integrate=integrate)


def MeasureStrehl(tel, Z, residual_phase):
    PSF_0 = PSF_from_KL(tel, residual_phase, diversity_phase=0, Z=Z, NCPA_OPDs=0, exposure_time=tel.det.sampling_time, residuals=False, randres=False)
    PSF_1 = PSF_from_KL(tel, residual_phase, diversity_phase=0, Z=Z, NCPA_OPDs=0, exposure_time=tel.det.sampling_time, residuals=True,  randres=True)
    print(PSF_1.max() / PSF_0.max())


Nph_diff = lambda m_1, m_2: 10**(-(m_2-m_1)/2.5)
mag_diff = lambda ph_diff: -2.5*np.log10(ph_diff)