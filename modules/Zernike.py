import sys
sys.path.insert(0, '..')

import numpy as np
try:
    import cupy as cp
    global_gpu_flag = True
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np
    global_gpu_flag = False

from tools.misc import mask_circle


class Zernike:
    def __init__(self, modes_num=1):
        global global_gpu_flag
        self.nModes = modes_num
        self.modesFullRes = None
        
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
            R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                            np.math.factorial(n-i)) / (np.math.factorial(i) *
                            np.math.factorial(int(0.5 * (n+m) - i)) *
                            np.math.factorial(int(0.5 * (n-m) - i))),
                            dtype='float')
        return R


    def zernIndex(self, j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n%2
        m = int((p+k)/2.)*2 - k

        if m!=0:
            if j % 2 == 0:  s=1
            else:  s=-1
            m *= s

        return [n, m]


    def computeZernike(self, tel, normalize_unit=False):
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
            pupil = mask_circle(N=resolution, r=resolution/2)
        else:
            pupil = tel.pupil.get() if self.gpu else tel.pupil

        X, Y = np.where(pupil == 1)
        X = (X-resolution//2+0.5*(1-resolution%2)) / resolution
        Y = (Y-resolution//2+0.5*(1-resolution%2)) / resolution
        R = np.sqrt(X**2 + Y**2)
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

            self.modesFullRes[np.where(np.reshape(pupil, resolution*resolution)>0), i-1] = Z
            
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
    def wavefrontFromModes(self, tel, coefs):
        xp = cp if self.gpu else np

        if isinstance(coefs, list): coefs = xp.array(coefs).flatten()

        if self.modesFullRes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.nModes = xp.max(xp.array([coefs.shape[0], self.nModes]))
            self.computeZernike(tel)

        if self.nModes < coefs.shape[0]:
            self.nModes = coefs.shape[0]
            print('Warning: vector of coefficients is too long. Computiong additional modes...')
            self.computeZernike(tel)

        phase = 0
        for i in range(coefs.shape[0]):
            if (coefs[i] is not None) and (not xp.isnan(coefs[i]).item()) and (xp.abs(coefs[i]).item()>1e-13):
                phase += self.modesFullRes[:,:,i] * coefs[i] * tel.pupil
        return phase


    def Mode(self, coef):
        return self.modesFullRes[:,:,coef]
