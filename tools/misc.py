import numpy as np
try:
    import cupy as cp
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np


# To make this function work, one must ensure that size of inp can be divided by N
def binning(inp, N):
    if N == 1: return inp
    xp = cp if hasattr(inp, 'device') else np
    out = xp.dstack(xp.split(xp.dstack(xp.split(inp, inp.shape[0]//N, axis=0)), inp.shape[1]//N, axis=1))
    return out.sum(axis=(0,1)).reshape([inp.shape[0]//N, inp.shape[1]//N])


def Gaussian2DTilted(amp=1.0, x_0=0.0, y_0=0.0, s_x=1.0, s_y=1.0, ang=0.0):
    if s_x < 1e-2 or s_y < 1e-2: return None

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


def NoisyPSF(tel, PSF, integrate=True):
    return tel.det.getFrame(PSF, noise=True, integrate=integrate)


Nph_diff = lambda m_1, m_2: 10**(-(m_2-m_1)/2.5)
mag_diff = lambda ph_diff: -2.5*np.log10(ph_diff)