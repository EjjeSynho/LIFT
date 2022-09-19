# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:31:33 2020

@author: cheritie
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:29:12 2020

@author: cheritie
"""

import numpy as np

class Zernike:
    def __init__(self, modes_num=1):
        self.nModes = modes_num
        #self.modes = None
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
                            np.math.factorial(n-i)) /
                            (np.math.factorial(i) *
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
            if j%2==0:
                s=1
            else:
                s=-1
            m *= s

        return [n, m]


    def computeZernike(self, tel):
        """
         Creates the Zernike polynomial with radial index, n, and azimuthal index, m.
    
         Args:
            n (int): The radial order of the zernike mode
            m (int): The azimuthal order of the zernike mode
            N (int): The diameter of the zernike more in pixels
         Returns:
            ndarray: The Zernike mode
         """
        X, Y = np.where(tel.pupil > 0)
        resolution = tel.pupil.shape[0]

        X = ( X-(resolution + resolution%2-1)/2 ) / resolution * tel.D
        Y = ( Y-(resolution + resolution%2-1)/2 ) / resolution * tel.D
        #                           ^- to properly allign coordinates relative to the (0,0) for even/odd telescope resolutions
        R = np.sqrt(X**2 + Y**2)
        R = R / R.max()
        theta = np.arctan2(Y, X)
        #self.modes = np.zeros( [tel.pupil.sum().astype('int'), self.nModes] )
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

            #self.modes[:, i-1] = Z
            self.modesFullRes[np.where(np.reshape(tel.pupil, resolution*resolution)>0), i-1] = Z
            
        self.modesFullRes = np.reshape( self.modesFullRes, [resolution, resolution, self.nModes] )


    def modeName(self, index):
        if index < 0:
            return('Incorrent index!')
        elif index >= len(self.modes_names):
            return('Z '+str(index+2))
        else:
            return(self.modes_names[index])


    # Generate wavefront shape corresponding to given model coefficients and modal basis 
    def wavefrontFromModes(self, tel, coefs):
        if isinstance(coefs, list):
            coefs = np.array(coefs)

        if self.modesFullRes is None:
            print('Warning: Zernike modes were not computed! Calculating...')
            self.nModes = np.max(np.array([coefs.shape[0], self.nModes]))
            self.computeZernike(tel)

        if self.nModes < coefs.shape[0]:
            self.nModes = coefs.shape[0]
            print('Warning: vector of coefficients is too long. Computiong additional modes...')
            self.computeZernike(tel)

        phase = 0
        for i in range(coefs.shape[0]):
            if (coefs[i] is not None) and (not np.isnan(coefs[i])) and (np.abs(coefs[i])>1e-13):
                phase += self.modesFullRes[:,:,i] * coefs[i]
        return phase

    def Mode(self, coef):
        return np.copy(self.modesFullRes[:,:,coef])
