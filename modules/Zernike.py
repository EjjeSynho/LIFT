import sys
sys.path.insert(0, '..')

import numpy as np
import math

try:
    import cupy as cp
    global_gpu_flag = True
except ImportError or ModuleNotFoundError:
    cp = np
    global_gpu_flag = False

from ..tools.misc import mask_circle


class Zernike:
    def __init__(self, modes_num=1):
        global global_gpu_flag
        self.nModes = modes_num
        self.modesFullRes = None
        self.pupil = None

        self.modes_names = [
            'Tip', 'Tilt', 'Defocus', 'Astigmatism (X)', 'Astigmatism (+)',
            'Coma vert', 'Coma horiz', 'Trefoil vert', 'Trefoil horiz',
            'Sphere', 'Secondary astig (X)', 'Secondary astig (+)',
            'Quadrofoil vert', 'Quadrofoil horiz',
            'Secondary coma horiz', 'Secondary coma vert',
            'Secondary trefoil horiz', 'Secondary trefoil vert',
            'Pentafoil horiz', 'Pentafoil vert'
        ]
        self.gpu = global_gpu_flag  


    @property
    def gpu(self):
        return self.__gpu

    @gpu.setter
    def gpu(self, var):
        if var:
            self.__gpu = True
            if hasattr(self, 'modesFullRes'):
                if not hasattr(self.modesFullRes, 'device'):
                    self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)
        else:
            self.__gpu = False
            if hasattr(self, 'modesFullRes'):
                if hasattr(self.modesFullRes, 'device'):
                    self.modesFullRes = self.modesFullRes.get()


    def zernikeRadialFunc(self, n, m, r):
        """
        Fucntion to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array

        Returns:
            ndarray: The Zernike radial function
        """

        R = np.zeros(r.shape)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n-m)/2) + 1):
            R += np.array(r**(n - 2 * i) * (((-1)**(i)) * \
                            math.factorial(n-i)) / (math.factorial(i) * \
                            math.factorial(int(0.5 * (n+m) - i)) * \
                            math.factorial(int(0.5 * (n-m) - i))), \
                            dtype='float')
        return R


    def zernIndex(self, j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0: s = 1
            else:  s = -1
            m *= s

        return [n, m]


    def rotate_coordinates(self, angle, X, Y):
            angle_rad = np.radians(angle)

            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])

            coordinates = np.vstack((X, Y))
            rotated_coordinates = np.dot(rotation_matrix, coordinates)
            rotated_X, rotated_Y = rotated_coordinates[0, :], rotated_coordinates[1, :]

            return rotated_X, rotated_Y
        

    def computeZernike(self, tel, normalize_unit=False, angle=None, transposed=False):
        """
        Function to calculate the Zernike modal basis

        Parameters:
            tel (Telescope): A telescope object, needed mostly to extract pupil data 
            normalize_unit (bool): Sets the regime for normalization of Zernike modes
                                   it's either the telescope's pupil or a unit circle  
        """
 
        resolution = tel.pupil.shape[0]

        self.gpu = self.gpu and tel.gpu
        if normalize_unit:
            self.pupil = mask_circle(N=resolution, r=resolution/2)
        else:
            self.pupil = tel.pupil.get() if self.gpu else tel.pupil

        X, Y = np.where(self.pupil == 1)
        X = (X-resolution//2+0.5*(1-resolution%2)) / resolution
        Y = (Y-resolution//2+0.5*(1-resolution%2)) / resolution
        
        if transposed:
            X, Y = Y, X
        
        if angle is not None and angle != 0.0:
            X, Y = self.rotate_coordinates(angle, X, Y)
        
        R = np.sqrt(X**2 + Y**2)
        R /= R.max()
        theta = np.arctan2(Y, X)

        self.modesFullRes = np.zeros([resolution**2, self.nModes])

        for i in range(1, self.nModes+1):
            n, m = self.zernIndex(i+1)
            if m == 0:
                Z = np.sqrt(n+1) * self.zernikeRadialFunc(n, 0, R)
            else:
                if m > 0: # j is even
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.cos(m*theta)
                else:   #i is odd
                    m = abs(m)
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.sin(m*theta)
            
            Z -= Z.mean()
            Z /= np.std(Z)

            self.modesFullRes[np.where(np.reshape(self.pupil, resolution*resolution)>0), i-1] = Z
            
        self.modesFullRes = np.reshape( self.modesFullRes, [resolution, resolution, self.nModes] )
        
        if self.gpu: # if GPU is used, return a GPU-based array
            self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)


    def modeName(self, index):
        if index < 0:
            return('Incorrent index!')
        elif index >= len(self.modes_names):
            return('Z ' + str(index+2))
        else:
            return(self.modes_names[index])


    # Generate wavefront shape corresponding to given model coefficients and modal basis 
    def wavefrontFromModes(self, tel, coefs_inp):
        xp = cp if self.gpu else np

        coefs = xp.array(coefs_inp).flatten()
        coefs[xp.where(xp.abs(coefs)<1e-13)] = xp.nan
        valid_ids = xp.where(xp.isfinite(coefs))[0]

        if self.modesFullRes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.nModes = xp.max(xp.array([coefs.shape[0], self.nModes]))
            self.computeZernike(tel)

        if self.nModes < coefs.shape[0]:
            self.nModes = coefs.shape[0]
            print('Warning: vector of coefficients is too long. Computiong additional modes...')
            self.computeZernike(tel)

        return self.modesFullRes[:,:,valid_ids] @ coefs[valid_ids] # * tel.pupil


    def Mode(self, coef):
        return self.modesFullRes[:,:,coef]


class CombinedModalBasis:
    """
    Lightweight modal-basis container produced by combineModalBases().

    It follows the same minimal interface as Zernike/LWE-like classes:
        - modesFullRes: [H, W, nModes]
        - nModes
        - modes_names
        - gpu
        - modeName(index)
        - wavefrontFromModes(tel, coefs_inp)
        - Mode(index)
    """

    def __init__(self, modesFullRes, modes_names=None, pupil=None, gpu=False):
        self.modesFullRes = modesFullRes
        self.nModes = modesFullRes.shape[-1]
        self.modes_names = list(modes_names) if modes_names is not None else [f'Mode {i}' for i in range(self.nModes)]
        self.pupil = pupil
        self.gpu = gpu

    def modeName(self, index):
        if index < 0:
            return('Incorrect index!')
        elif index >= len(self.modes_names):
            return('Mode ' + str(index))
        else:
            return(self.modes_names[index])

    def wavefrontFromModes(self, tel, coefs_inp):
        xp = cp if self.gpu else np

        coefs = xp.array(coefs_inp).flatten()
        coefs[xp.where(xp.abs(coefs) < 1e-13)] = xp.nan
        valid_ids = xp.where(xp.isfinite(coefs))[0]

        if self.nModes < coefs.shape[0]:
            raise ValueError(
                f'Coefficient vector has length {coefs.shape[0]}, '
                f'but combined basis has only {self.nModes} modes.'
            )

        return self.modesFullRes[:, :, valid_ids] @ coefs[valid_ids]

    def Mode(self, coef):
        return self.modesFullRes[:, :, coef]


def _modal_basis_array(modal_basis):
    """
    Return modal basis array from either a modal-basis object or a raw array.

    Expected convention is [H, W, nModes].
    """
    if hasattr(modal_basis, 'modesFullRes'):
        modes = modal_basis.modesFullRes
    else:
        modes = modal_basis

    if modes is None:
        raise ValueError('One of the supplied modal bases has modesFullRes=None. Compute it first.')

    if len(modes.shape) != 3:
        raise ValueError(f'Expected modal basis shape [H, W, nModes], got {modes.shape}.')

    return modes


def _modal_basis_names(modal_basis, prefix='Mode'):
    """
    Extract mode names from an object if possible; otherwise generate generic names.
    """
    modes = _modal_basis_array(modal_basis)
    n_modes = modes.shape[-1]

    if hasattr(modal_basis, 'modeName'):
        return [modal_basis.modeName(i) for i in range(n_modes)]

    if hasattr(modal_basis, 'modes_names') and modal_basis.modes_names is not None:
        names = list(modal_basis.modes_names)
        if len(names) >= n_modes:
            return names[:n_modes]

    return [f'{prefix} {i}' for i in range(n_modes)]


def _modal_basis_is_gpu(modal_basis):
    if hasattr(modal_basis, 'gpu'):
        return bool(modal_basis.gpu)

    modes = _modal_basis_array(modal_basis)
    return hasattr(modes, 'device')


def _to_backend_array(array, use_gpu):
    """
    Move array to NumPy/CuPy backend requested by use_gpu.
    """
    if use_gpu:
        return cp.array(array, dtype=cp.float32)

    if hasattr(array, 'get'):
        return array.get()

    return np.asarray(array)


def combineModalBases(
    zernike_basis,
    lwe_basis,
    lwe_position='before',
    inplace=False,
    zernike_prefix='Zernike',
    lwe_prefix='LWE'
):
    """
    Concatenate Zernike and LWE modal bases.

    Parameters
    ----------
    zernike_basis : object or ndarray
        Zernike-like modal-basis object with `.modesFullRes`, or a raw array
        with shape [H, W, nZernike].

    lwe_basis : object or ndarray
        LWE-like modal-basis object with `.modesFullRes`, or a raw array
        with shape [H, W, nLWE].

    lwe_position : {'before', 'after'}, optional
        If 'before', output basis is [LWE, Zernike].
        If 'after',  output basis is [Zernike, LWE].

    inplace : bool, optional
        If True and `zernike_basis` is an object, modify it in-place and return it.
        If False, return a CombinedModalBasis object.

    zernike_prefix, lwe_prefix : str, optional
        Prefixes used in generated / decorated mode names.

    Returns
    -------
    CombinedModalBasis or zernike_basis
        Combined modal-basis container. Its `.modesFullRes` has shape
        [H, W, nZernike + nLWE].

    Examples
    --------
    >>> zern = Zernike(20)
    >>> zern.computeZernike(tel)
    >>> lwe = LWE()
    >>> lwe.computeLWE(tel)
    >>> modes = combineModalBases(zern, lwe, lwe_position='before')
    >>> wf = modes.wavefrontFromModes(tel, coefs)
    """
    if lwe_position not in ('before', 'after'):
        raise ValueError("lwe_position must be either 'before' or 'after'.")

    z_modes = _modal_basis_array(zernike_basis)
    l_modes = _modal_basis_array(lwe_basis)

    if z_modes.shape[:2] != l_modes.shape[:2]:
        raise ValueError(
            'Zernike and LWE bases must have the same spatial resolution. '
            f'Got {z_modes.shape[:2]} and {l_modes.shape[:2]}.'
        )

    use_gpu = _modal_basis_is_gpu(zernike_basis) or _modal_basis_is_gpu(lwe_basis)
    z_modes = _to_backend_array(z_modes, use_gpu)
    l_modes = _to_backend_array(l_modes, use_gpu)
    xp = cp if use_gpu else np

    z_names = [f'{zernike_prefix}: {name}' for name in _modal_basis_names(zernike_basis, zernike_prefix)]
    l_names = [f'{lwe_prefix}: {name}' for name in _modal_basis_names(lwe_basis, lwe_prefix)]

    if lwe_position == 'before':
        combined_modes = xp.concatenate([l_modes, z_modes], axis=-1)
        combined_names = l_names + z_names
    else:
        combined_modes = xp.concatenate([z_modes, l_modes], axis=-1)
        combined_names = z_names + l_names

    pupil = getattr(zernike_basis, 'pupil', None)
    if pupil is None:
        pupil = getattr(lwe_basis, 'pupil', None)

    if inplace:
        if not hasattr(zernike_basis, 'modesFullRes'):
            raise ValueError('inplace=True requires zernike_basis to be an object with modesFullRes.')

        zernike_basis.modesFullRes = combined_modes
        zernike_basis.nModes = combined_modes.shape[-1]
        zernike_basis.modes_names = combined_names
        zernike_basis.pupil = pupil
        zernike_basis.gpu = use_gpu
        return zernike_basis

    return CombinedModalBasis(
        modesFullRes=combined_modes,
        modes_names=combined_names,
        pupil=pupil,
        gpu=use_gpu
    )
