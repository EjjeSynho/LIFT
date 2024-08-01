import sys
sys.path.insert(0, '..')

import numpy as np
from scipy import signal as sg
import warnings

global global_gpu_flag

try:
    import cupy as cp
    from cupyx.scipy import signal as csg

    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    cp  = np
    csg = sg
    global_gpu_flag = False

from tools.misc import binning


class LIFT:
    def __init__(self, tel, modeBasis, diversity_OPD, iterations):
        self.tel             = tel
        self.modeBasis       = modeBasis
        self.diversity_OPD   = diversity_OPD
        self.iterations      = iterations
        self.gpu             = self.tel.gpu and global_gpu_flag

    @property
    def gpu(self):
        return self.__gpu

    @gpu.setter
    def gpu(self, var):
        if var:
            self.__gpu = True
            if hasattr(self, 'diversity_OPD'):
                if not hasattr(self.diversity_OPD, 'device'):
                    self.diversity_OPD = cp.array(self.diversity_OPD, dtype=cp.float32)
        else:
            self.__gpu = False
            if hasattr(self, 'diversity_OPD'):
                if hasattr(self.diversity_OPD, 'device'):
                    self.diversity_OPD = self.diversity_OPD.get()


    def print_modes(self, A_vec):
        xp = cp if self.gpu else np
        for i in range(A_vec.shape[0]):
            val = A_vec[i]
            if val != None and not xp.isnan(val): val = xp.round(val, 4)
            print('Mode #', i, val)


    def obj_convolve(self, mat):
        xg = csg if self.gpu else sg
        if self.tel.object is not None:
            return xg.convolve2d(mat, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()
        else: return mat


    def __check_modes(self, modes):
        for mode in modes:
            if mode < 0:
                warnings.warn('Negative mode number detected, this is not allowed. Removed.')
                modes.remove(mode)
            if mode >= self.modeBasis.modesFullRes.shape[2]:
                warnings.warn('Mode number exceeds the number of modes in the basis. Removed. Consider regenerating the modal basis with more modes included.')
                modes.remove(mode)        
        return modes


    def generateLIFTinteractionMatrices(self, coefs, modes_ids, flux_norm=1.0, numerical=False):
        xp = cp if self.gpu else np

        if isinstance(coefs, list):
            coefs = xp.array(coefs)

        initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs) + self.diversity_OPD
        H = []
        
        # Analytical interaction matrices
        if not numerical:
            for point in self.tel.src.spectrum:
                wavelength = point['wavelength']

                initial_amplitude = xp.sqrt(self.tel.flux(point['flux']*flux_norm, self.tel.det.sampling_time)) * self.tel.pupil
                k = 2*xp.pi/wavelength

                initial_phase = k * initial_OPD
                Pd = xp.conj( self.tel.PropagateField(initial_amplitude, initial_phase, wavelength, return_intensity=False) )
                
                H_spectral = []
                for i in modes_ids:
                    buf = self.tel.PropagateField(self.modeBasis.modesFullRes[:,:,i]*initial_amplitude, initial_phase, wavelength, return_intensity=False)
                    derivative = 2*binning((xp.real(1j*buf*Pd)), self.tel.oversampling) * k
                    derivative = self.obj_convolve(derivative)

                    H_spectral.append(derivative.flatten())   
                H.append(xp.vstack(H_spectral).T)
        
        # Numerical interaction matrices
        else:
            delta = 1e-9 # [nm]
            for point in self.tel.src.spectrum:
                H_spectral = []
                for i in modes_ids:
                    self.tel.src.OPD = (self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                    tmp1 = self.tel.ComputePSF() * flux_norm           

                    self.tel.src.OPD = -(self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                    tmp2 = self.tel.ComputePSF() * flux_norm

                    derivative = self.obj_convolve((tmp1-tmp2)/2/delta)

                    H_spectral.append( derivative.flatten() )
                H.append(np.vstack(H_spectral).T)
        
        return xp.dstack(H).sum(axis=2) # sum all spectral interaction matricies


    def Reconstruct(self, PSF_inp, R_n, mode_ids, A_0=None, verbous=False, optimize_norm='sum'):
        """
        Iterative Maximum-Likelihood estimation of coefficients from the input PSF image using LIFT.

        Parameters:
            PSF (ndarray):                   2-d array of the input PSF image to reconstruct.

            R_n (ndarray or string or None): The pixel weighting matrix for LIFT. It can be passed to the function.
                                             from outside, modeled ('model'), updated dynamically ('iterative'), or
                                             assumed to be just detector's readout noise ('None').
                                             
            mode_ids (ndarray or list):      IDs of the modal coefficients to be reconstructed.
            
            A_0 (ndarray):                   Initial assumption for the coefficient values. Acts like a starting point for the optimization.
            
            verbous (bool):                  Set 'True' to print the intermediate reconstruction results.

            optimize_norm (string or None):  Recomputes the flux of the recontructed PSF iteratively. If 'None', the flux is not recomputed, 
                                             this is recommended only if the target brightness is precisely known. In mosyt of the case it is 
                                             recommended to switch it on. When 'sum', the reconstructed PSF is normalized to the sum of the pixels 
                                             of the input PSF. If 'max', then the reconstructed PSF is normalized to the maximal value of the input PSF.
        """
        if self.gpu:
            xp = cp
            convert = lambda x: cp.asnumpy(x)
        else:
            xp = np
            convert = lambda x: x

        def PSF_from_coefs(coefs):
            OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs)
            self.tel.src.OPD = self.diversity_OPD + OPD
            return self.obj_convolve( self.tel.ComputePSF() )

        mode_ids = self.__check_modes(mode_ids)
        
        C      = []  # optimization criterion
        Hs     = []  # interaction matrices for every iteration
        P_MLs  = []  # estimators for every iteration
        A_ests = []  # history of estimated coefficients
        
        PSF   = xp.array(PSF_inp, dtype=xp.float32)
        modes = xp.sort( xp.array(mode_ids, dtype=xp.int32) )

        # Account for the intial assumption for the coefficients values
        if A_0 is None:
            A_est = xp.zeros(modes.max().item()+1, dtype=xp.float32)
        else:
            A_est = xp.array(A_0, dtype=xp.float32)
        A_ests.append(xp.copy(A_est))

        def normalize_PSF(PSF_in):
            if optimize_norm is not None and optimize_norm is not False:
                if optimize_norm == 'max': return (PSF_in/PSF_in.max(), PSF_in.max())
                if optimize_norm == 'sum': return (PSF_in/PSF_in.sum(), PSF_in.sum())
            else: return (PSF_in, 1.0)
        
        PSF_0, flux_cap = normalize_PSF(PSF_from_coefs(A_est)) # initial PSF assumption, normalized to 1.0

        flux_scale = 1.0
        if optimize_norm is not None and optimize_norm is not False:
            if   optimize_norm == 'max': flux_scale = PSF.max()
            elif optimize_norm == 'sum': flux_scale = PSF.sum()
        
        PSF_cap = xp.copy(PSF_0) * flux_scale

        criterion  = lambda i: xp.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: xp.linalg.norm(v[modes], ord=2)

        def inverse_Rn(Rn):
            return 1./xp.clip(Rn.flatten(), a_min=1e-6, a_max=Rn.max())  

        if R_n is not None:
            if isinstance(R_n, str): #basically if it's 'model' or 'iterative':
                inv_R_n = inverse_Rn(PSF_0*flux_scale + self.tel.det.readoutNoise**2)
            else:
                inv_R_n = inverse_Rn(xp.array(R_n))
        else:
            inv_R_n = inverse_Rn(xp.ones_like(PSF)*self.tel.det.readoutNoise**2)

        for i in range(self.iterations):    
            dI = (PSF-PSF_cap).flatten()
    
            C.append( xp.dot(dI*inv_R_n, dI) )  # check the convergence
            if i > 0 and (criterion(i)<1e-5 or coefs_norm(A_ests[i]-A_ests[i-1])<10e-9):
                if verbous:
                    print('Criterion', criterion(i), 'is reached at iter.', i)
                break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(A_est, modes, flux_scale/flux_cap)                               

            # Maximum likelyhood estimation
            P_ML = xp.linalg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            d_A = P_ML @ dI
            A_est[modes] += d_A
            
            # Save the intermediate results for history
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(xp.copy(A_est))

            if verbous:
                print('Criterion:', criterion(i))
                self.print_modes(d_A)
                print()
            
            # Update the PSF image with the estimated coefficients
            PSF_cap, flux_cap = normalize_PSF(PSF_from_coefs(A_est))
            PSF_cap *= flux_scale
            
            if isinstance(R_n, str):
                if R_n == 'iterative':
                    inv_R_n = inverse_Rn(PSF_cap + self.tel.det.readoutNoise**2)

        history = { # contains intermediate data saved at every iteration
            'P_ML' : convert( xp.dstack(P_MLs) ),
            'H'    : convert( xp.dstack(Hs) ),
            'A_est': convert( xp.squeeze(xp.dstack(A_ests), axis=0) ),
            'C'    : convert( xp.array(C) )
        }
        return convert(A_est), convert(PSF_cap), history


    def ReconstructMAP(self, PSF_inp, R_n, mode_ids, A_mean, A_var, A_0=None, verbous=False, optimize_norm='sum'):
        """
        Iterative Maximum A posteriori Probablity estimation of coefficients from the input PSF image using LIFT.

        Parameters:
            PSF (ndarray):                   2-d array of the input PSF image to reconstruct.

            R_n (ndarray or string or None): The pixel weighting matrix for LIFT. It can be passed to the function.
                                             from outside, modeled ('model'), updated dynamically ('iterative'), or
                                             assumed to be just detector's readout noise ('None').
                                             
            mode_ids (ndarray or list):      IDs of the modal coefficients to be reconstructed
            
            A_mean (ndarray):                Expected values of coefficients.

            A_var (ndarray):                 Variance of coefficients.
            
            A_0 (ndarray):                   Initial assumption for the coefficient values. Acts like a starting point for the optimization.

            verbous (bool):                  Set 'True' to print the intermediate reconstruction results.

            optimize_norm (string or None):  Recomputes the flux of the recontructed PSF iteratively. If 'None', the flux is not recomputed, 
                                             this is recommended only if the target brightness is precisely known. In mosyt of the case it is 
                                             recommended to switch it on. When 'sum', the reconstructed PSF is normalized to the sum of the pixels 
                                             of the input PSF. If 'max', then the reconstructed PSF is normalized to the maximal value of the input PSF.
        """
        if self.gpu:
            try:
                xp = cp
                convert = lambda x: cp.asnumpy(x)
            except:
                xp = np
                convert = lambda x: x
        else:
            xp = np
            convert = lambda x: x
        
        mode_ids = self.__check_modes(mode_ids)
        
        # Check device of the input data
        def check_backend(data):
            if data is None:
                return None
            
            if not isinstance(data, xp.ndarray):
                warnings.warn('Wrong backend of the input data, converting to the proper one...')
                return xp.array(data, dtype=xp.float32).squeeze() # avoid adding singleton dimensions
            else:
                return data
                
        PSF_inp = check_backend(PSF_inp)
        R_n     = check_backend(R_n)
        A_mean  = check_backend(A_mean)
        A_var   = check_backend(A_var)
        A_0     = check_backend(A_0)

        def PSF_from_coefs(coefs):
            OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs)
            self.tel.src.OPD = self.diversity_OPD + OPD
            return self.obj_convolve( self.tel.ComputePSF() )
                   
        C      = []  # optimization criterion
        Hs     = []  # interaction matrices for every iteration
        A_ests = []  # history of estimated coefficients
        
        PSF   = xp.array(PSF_inp, dtype=xp.float32)
        modes = xp.sort( xp.array(mode_ids, dtype=xp.int32) )

        # Account for the intial assumption for the coefficients values
        if A_0 is None or xp.any(A_0[mode_ids] == np.nan):
            A_est = xp.zeros(modes.max().item()+1, dtype=xp.float32)
        else:
            A_est = xp.array(A_0, dtype=xp.float32)
        
        A_ests.append(xp.copy(A_est))

        C_phi_inv = 1.0 / A_var[mode_ids] # Inverse of the covariance vector
        A_ = A_mean[mode_ids]
        
        def normalize_PSF(PSF_in):
            if optimize_norm is not None and optimize_norm is not False:
                if optimize_norm == 'max': return (PSF_in/PSF_in.max(), PSF_in.max())
                if optimize_norm == 'sum': return (PSF_in/PSF_in.sum(), PSF_in.sum())
            else: return (PSF_in, 1.0)
        
        PSF_0, flux_cap = normalize_PSF(PSF_from_coefs(A_est)) # initial PSF assumption, normalized to 1.0

        flux_scale = 1.0
        if optimize_norm is not None and optimize_norm is not False:
            if   optimize_norm == 'max': flux_scale = PSF.max()
            elif optimize_norm == 'sum': flux_scale = PSF.sum()
        
        PSF_cap = xp.copy(PSF_0) * flux_scale

        criterion  = lambda i: xp.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: xp.linalg.norm(v[modes], ord=2)

        def inverse_Rn(Rn):
            return 1./xp.clip(Rn.flatten(), a_min=1e-6, a_max=Rn.max())  

        if R_n is not None:
            if isinstance(R_n, str): #basically if it's 'model' or 'iterative':
                inv_R_n = inverse_Rn(PSF_0*flux_scale + self.tel.det.readoutNoise**2)
            else:
                inv_R_n = inverse_Rn(xp.array(R_n))
        else:
            inv_R_n = inverse_Rn(xp.ones_like(PSF)*self.tel.det.readoutNoise**2)


        for i in range(self.iterations):    
            dI = (PSF-PSF_cap).flatten()
    
            C.append( xp.dot(dI*inv_R_n, dI) )  # check the convergence
            if i > 0 and (criterion(i)<1e-5 or coefs_norm(A_ests[i]-A_ests[i-1])<10e-9):
                if verbous:
                    print('Criterion', criterion(i), 'is reached at iter.', i)
                break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(A_est, modes, flux_scale/flux_cap)                               

            # Maximum likelyhood estimation
            d_A = xp.linalg.pinv(H.T * inv_R_n @ H + C_phi_inv) @ ( (H.T * inv_R_n) @ dI + C_phi_inv * A_ )
            A_est[modes] += d_A
            
            # Save the intermediate results for history
            Hs.append(H)
            A_ests.append(xp.copy(A_est))

            if verbous:
                print('Criterion:', criterion(i))
                self.print_modes(d_A)
                print()
            
            # Update the PSF image with the estimated coefficients
            PSF_cap, flux_cap = normalize_PSF(PSF_from_coefs(A_est))
            PSF_cap *= flux_scale
            
            if isinstance(R_n, str):
                if R_n == 'iterative':
                    inv_R_n = inverse_Rn(PSF_cap + self.tel.det.readoutNoise**2)

        history = { # contains intermediate data saved at every iteration
            'H'    : convert( xp.dstack(Hs) ),
            'A_est': convert( xp.squeeze(xp.dstack(A_ests), axis=0) ),
            'C'    : convert( xp.array(C) )
        }
        return convert(A_est), convert(PSF_cap), history


    def __ReconstructLegacy(self, PSF, R_n, mode_ids, A_0=None, A_ref=None, verbous=False, optimize_norm=False):    
        if self.gpu:
            xsg = csg
            xp  = cp
        else:
            xsg = sg
            xp  = np
        
        # Since coefficient vector A can contain Nones, they must be ignored during the calculation
        def pack(A_vec):
            return xp.array( [a_k.item() for a_k in A_vec if a_k.item() != None and not xp.isnan(a_k).item()] )

        # Back to the form where position of the coeeficient in the vector correspond to the number of  mode
        def unpack(coef_vec, mode_nums):
            A_vec = [xp.nan for _ in range(max(mode_nums)+1)]
            for i, mode in enumerate(mode_nums):
                A_vec[mode] = coef_vec[i].item()
            return xp.array(A_vec)
            
        C  = []         # optimization criterion
        Hs = []         # interaction matrices for every iteration
        P_MLs = []      # estimators for every iteration
        A_ests = []     # history of estimated coefficients
        
        mode_ids.sort()

        # If coefficients ground truth is known
        if A_ref is not None:
            if xp.all(xp.abs(pack(A_ref)) < 1e-9):
                print('Warning: nothing to optimize, all ref. coefs. = 0')
                return A_ref, None, None

        # Account for the initial approximation for aberrations
        initial_OPD = 0.0
        A_est = xp.zeros(max(mode_ids)+1)

        if A_0 is not None:
            modes_0 = np.where(np.isfinite(A_0))[0].tolist() # modes defined in the initial vector
            for i in set(mode_ids) & set(modes_0):
                A_est[i] = A_0[i]
            initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, A_0)

        for i in set(range(max(mode_ids)+1)) - set(mode_ids):
            A_est[i] = xp.nan
        A_est = pack(A_est)

        self.tel.src.OPD = self.diversity_OPD + initial_OPD
        PSF_0 = self.tel.ComputePSF()
    
        # Take a priory object information into account
        if self.tel.object is not None:
            PSF_0 = csg.convolve2d(PSF_0, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        A_ests.append(xp.array(A_est)) # saving coefficients history
        PSF_cap = xp.array(PSF_0)      # saving PSF history
        inv_R_n = 1.0 / xp.clip(R_n.reshape(R_n.shape[0]*R_n.shape[1]), a_min=1e-6, a_max=R_n.max())

        criterion  = lambda i: xp.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: xp.linalg.norm(v, ord=2)

        for i in range(self.iterations):
            PSF_buf = xp.array(PSF)
            inv_R_n_buf = xp.array(inv_R_n)

            if optimize_norm is not None:
                if optimize_norm == 'max':
                    flux_scale = PSF_cap.max() / PSF.max()
                elif optimize_norm == 'sum':
                    flux_scale = PSF_cap.sum() / PSF.sum()
                else: flux_scale = 1.0

                PSF_buf *= flux_scale
                inv_R_n_buf = inv_R_n_buf / flux_scale

            dI = (PSF_buf - PSF_cap).flatten().astype('float64')

            # Check convergence
            C.append( xp.dot(dI * inv_R_n_buf, dI) )
            
            if i > 0 and (criterion(i) < 1e-6 or coefs_norm(A_ests[i]-A_ests[i-1]) < 1e-12):
                if verbous: print('Criterion', criterion(i), 'is reached at iter.', i)
                break

            if A_ref is not None:
                if coefs_norm(pack(A_ref)-A_est) < 1e-12:
                    print('Err. vec. norm', coefs_norm(pack(A_ref)-A_est), 'is reached at iter.', i)
                    break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(
                coefs = unpack(A_est, mode_ids),
                modes_ids = mode_ids,
                flux_norm = 1.0,
                numerical = False
            )

            # Maximum likelyhood estimation
            #P_ML = lg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            P_ML = xp.linalg.pinv(H.T * inv_R_n_buf @ H) @ H.T * inv_R_n_buf
            d_A = P_ML @ dI
            A_est += d_A
            
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(xp.copy(A_est))

            if verbous:
                print( 'Iteration:', i )  
                if A_ref is not None:
                    print( 'Criterion:', criterion(i), ', err. vec. norm:', coefs_norm(pack(A_ref)*1e9-A_est*1e9))
                else:
                    print( 'Criterion:', criterion(i))
                self.print_modes(unpack(d_A*1e9, mode_ids))
                print()
            
            # Recreate the spot with reconstructed coefficients
            estimated_OPD = self.modeBasis.wavefrontFromModes(self.tel, unpack(A_est, mode_ids))
            self.tel.src.OPD = estimated_OPD + self.diversity_OPD
            PSF_cap = self.tel.ComputePSF()

            if self.tel.object is not None:
                PSF_cap = xsg.convolve2d(PSF_cap, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

            if optimize_norm is not None:
                if optimize_norm == 'max':
                    PSF_cap = PSF_cap * PSF.max() / PSF_cap.max()
                elif optimize_norm == 'sum':
                    PSF_cap = PSF_cap * PSF.sum() / PSF_cap.sum()
                else: pass
                
        if self.gpu:
            history = { #save intermediate data every iteration
                    'P_ML' : xp.dstack(P_MLs).get(),
                    'H'    : xp.dstack(Hs).get(),
                    'A_est': xp.squeeze( xp.dstack(A_ests), axis=0 ).get(),
                    'C'    : xp.array(C).get()
                }
            return unpack(A_est, mode_ids).get(), PSF_cap.get(), history
        
        else:
            history = { #save intermediate data every iteration
                'P_ML' : xp.dstack(P_MLs),
                'H'    : xp.dstack(Hs),
                'A_est': xp.squeeze( xp.dstack(A_ests), axis=0 ),
                'C'    : xp.array(C)
            }
            return unpack(A_est, mode_ids), PSF_cap, history


    def ReconstructLegacy(self, PSF, R_n, mode_ids, A_0=None, A_ref=None, verbous=False, optimize_norm='sum'):
        """
        Old version of the function to reconstruct modal coefficients from the input PSF image using LIFT

        Parameters:
            PSF (ndarray):                   2-d array of the input PSF image to reconstruct.

            R_n (ndarray or string or None): The pixel weighting matrix for LIFT. It can be passed to the function.
                                             from outside, modeled ('model'), or assumed to be unitary ('None').
            mode_ids (ndarray or list):      IDs of the modal coefficients to be reconstructed
            
            A_0 (ndarray):                   Initial assumption for the coefficient values. Acts like a starting point for the optimization.
            
            A_ref (ndarray):                 Reference coefficients to compare the reconstruction with. Useful only when ground-truth (A_ref) is known.

            verbous (bool):                  Set 'True' to print the intermediate reconstruction results.

            optimize_norm (string or None):  Recomputes the flux of the recontructed PSF iteratively. If 'None', the flux is not recomputed, 
                                             this is recommended only if the target brightness is precisely known. In mosyt of the case it is 
                                             recommended to switch it on. When 'sum', the reconstructed PSF is normalized to the sum of the pixels 
                                             of the input PSF. If 'max', then the reconstructed PSF is normalized to the maximal value of the input PSF.
        """

        xp = cp if self.gpu else np

        if type(R_n) == str:
            if R_n == 'model':
                self.tel.src.OPD = self.tel.pupil
                PSF_model = self.tel.ComputePSF()

                if optimize_norm is not None:
                    if optimize_norm == 'sum':
                        PSF_model =  PSF_model / PSF_model.sum() * PSF.sum()
                    elif optimize_norm == 'max':
                        PSF_model =  PSF_model / PSF_model.max() * PSF.max()
                    else:
                        pass
                    
                R_n = PSF_model + self.tel.det.readoutNoise
                
        elif R_n is None:
            R_n = xp.ones_like(PSF)
            
        else:
            pass                            

        return self.__ReconstructLegacy(PSF, R_n, mode_ids, A_0, A_ref, verbous, optimize_norm)