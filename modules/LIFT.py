import numpy as np
from scipy import linalg as lg
from scipy import signal as sg

try:
    import cupy as cp
    from cupy import linalg as clg
    from cupyx.scipy import signal as csg
    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp  = np
    csg = sg
    clg = lg
    global_gpu_flag = False


from tools.misc import binning

class LIFT:
    def __init__(self, tel, modeBasis, astigmatism_OPD, iterations):
        global global_gpu_flag
        self.tel             = tel
        self.modeBasis       = modeBasis
        self.astigmatism_OPD = astigmatism_OPD
        self.iterations      = iterations
        self.gpu             = self.tel.gpu and global_gpu_flag

        if self.gpu:
            self.astigmatism_OPD = cp.array(self.astigmatism_OPD, dtype=cp.float32)


    # Since coefficient vector A can contain Nones, they must be ignored during the calculation
    def pack(self, A_vec):
        xp = cp if self.gpu else np
        return xp.array( [a_k.item() for a_k in A_vec if a_k.item() != None and not xp.isnan(a_k).item()] )


    # Back to the form where position of the coeeficient in the vector correspond to the number of  mode
    def unpack(self, coef_vec, mode_nums):
        xp = cp if self.gpu else np        
        A_vec = [xp.nan for _ in range(max(mode_nums)+1)]
        for i, mode in enumerate(mode_nums):
            A_vec[mode] = coef_vec[i].item()
        return xp.array(A_vec)


    def print_modes(self, A_vec):
        xp = cp if self.gpu else np
        for i in range(A_vec.shape[0]):
            val = A_vec[i]
            if val != None and not xp.isnan(val): val = xp.round(val, 4)
            print('Mode #', i, val)


    def generateLIFTinteractionMatrices(self, coefs, numerical=False):
        xp = cp  if self.gpu else np
        xg = csg if self.gpu else sg

        if isinstance(coefs, list):  coefs = xp.array(coefs)

        if self.gpu:
            if self.tel.object is not None:
                self.tel.object = cp.array(self.tel.object, cp.float32)

        initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs) + self.astigmatism_OPD
        
        H = []
        if not numerical:
            for point in self.tel.src.spectrum:
                wavelength = point['wavelength']

                initial_amplitude = xp.sqrt(self.tel.flux(point['flux'], self.tel.det.sampling_time)) * self.tel.pupil
                k = 2*xp.pi/wavelength

                initial_phase = k * initial_OPD
                Pd = xp.conj( self.tel.PropagateField(initial_amplitude, initial_phase, wavelength, return_intensity=False) )
                
                H_spectral = []
                for i in range(coefs.shape[0]):
                    if coefs[i].item() is not None and not xp.isnan(coefs[i]).item():
                        buf = self.tel.PropagateField(self.modeBasis.modesFullRes[:,:,i]*initial_amplitude, initial_phase, wavelength, return_intensity=False)
                        derivative = 2*binning((xp.real(1j*buf*Pd)), self.tel.oversampling) * k

                        if self.tel.object is not None:
                            derivative = xg.convolve2d(derivative, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

                        H_spectral.append(derivative.flatten())   
                H.append(xp.vstack(H_spectral).T)
        
        else:
            delta = 1e-9 # [nm]
            for point in self.tel.src.spectrum:
                H_spectral = []
                for i in range(coefs.shape[0]):
                    if (coefs[i] is not None) and (not np.isnan(coefs[i])):       
                        self.tel.src.OPD = (self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                        tmp1 = self.tel.ComputePSF()                

                        self.tel.src.OPD = -(self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                        tmp2 = self.tel.ComputePSF()


                        derivative = (tmp1-tmp2) / 2 / delta
                            
                        if self.tel.object is not None:
                            derivative = xg.convolve2d(derivative, self.tel.object, boundary='symm', mode='same')
                        
                        H_spectral.append( derivative.flatten() )
                H.append(np.vstack(H_spectral).T)
        
        return xp.dstack(H).sum(axis=2) # sum all spectral interaction matricies


    def ReconstructCPU(self, PSF, R_n, mode_ids, A_0=None, A_ref=None, verbous=False, optimize_norm=False):
        C  = []         # optimization criterion
        Hs = []         # interaction matrices for every iteration
        P_MLs = []      # estimators for every iteration
        A_ests = []     # history of estimated coefficients
        
        mode_ids.sort()

        # If coefficients ground truth is known
        if A_ref is not None:
            if np.all(np.abs(self.pack(A_ref)) < 1e-9):
                print('Warning: nothing to optimize, all ref. coefs. = 0')
                return A_ref, None, None

        # Account for the initial approximation for aberrations
        initial_OPD = 0.0
        A_est = np.zeros(max(mode_ids)+1)

        if A_0 is not None:
            modes_0 = np.where(np.isfinite(A_0))[0].tolist() # modes defined in the initial vector
            for i in set(mode_ids) & set(modes_0):
                A_est[i] = A_0[i]
            initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, A_0)

        for i in set(range(max(mode_ids)+1)) - set(mode_ids):
            A_est[i] = np.nan
        A_est = self.pack(A_est)

        self.tel.src.OPD = self.astigmatism_OPD + initial_OPD
        PSF_0 = self.tel.ComputePSF()
    
        # Take a priory object information into account
        if self.tel.object is not None:
            PSF_0 = sg.convolve2d(PSF_0, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        A_ests.append(np.copy(A_est)) # saving coefficients history
        PSF_cap = np.copy(PSF_0)      # saving PSF history
        inv_R_n = 1.0 / np.clip(R_n.reshape(R_n.shape[0]*R_n.shape[1]), a_min=1e-6, a_max=R_n.max())

        criterion  = lambda i: np.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: np.linalg.norm(v, ord=2)

        for i in range(self.iterations):
            PSF_buf = np.copy(PSF)
            inv_R_n_buf = np.copy(inv_R_n)

            if optimize_norm:
                flux_scale = PSF_cap.max() / PSF.max()
                PSF_buf *= flux_scale
                inv_R_n_buf = inv_R_n_buf / flux_scale**2 #because it's a variance

            dI = (PSF_buf - PSF_cap).flatten().astype('float64')

            # Check convergence
            #C.append( np.dot(dI * inv_R_n, dI) )
            C.append( np.dot(dI * inv_R_n_buf, dI) )
            
            if i > 0 and (criterion(i) < 1e-6 or coefs_norm(A_ests[i]-A_ests[i-1]) < 1e-12):
                print('Criterion', criterion(i), 'is reached at iter.', i)
                break

            if A_ref is not None:
                if coefs_norm(self.pack(A_ref)-A_est) < 1e-12:
                    print('Err. vec. norm', coefs_norm(self.pack(A_ref)-A_est), 'is reached at iter.', i)
                    break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(coefs=self.unpack(A_est, mode_ids))                               

            # Maximum likelyhood estimation
            #P_ML = lg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            P_ML = lg.pinv(H.T * inv_R_n_buf @ H) @ H.T * inv_R_n_buf
            d_A = P_ML @ dI
            A_est += d_A
            
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(np.copy(A_est))

            if verbous:
                print( 'Iteration:', i )  
                if A_ref is not None:
                    print( 'Criterion:', criterion(i), ', err. vec. norm:', coefs_norm(self.pack(A_ref)*1e9-A_est*1e9))
                else:
                    print( 'Criterion:', criterion(i))
                self.print_modes(self.unpack(d_A*1e9, mode_ids))
                print()
            
            # Recreate the spot with reconstructed coefficients
            estimated_OPD = self.modeBasis.wavefrontFromModes(self.tel, self.unpack(A_est, mode_ids))
            self.tel.src.OPD = estimated_OPD + self.astigmatism_OPD
            PSF_cap = self.tel.ComputePSF()

            if self.tel.object is not None:
                PSF_cap = sg.convolve2d(PSF_cap, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        if optimize_norm:
            PSF_cap = PSF_cap * PSF.max() / PSF_cap.max()

        history = { #save intermediate data every iteration
                'P_ML' : np.dstack(P_MLs),
                'H'    : np.dstack(Hs),
                'A_est': np.squeeze( np.dstack(A_ests), axis=0 ),
                'C'    : np.array(C)
            }
        return self.unpack(A_est, mode_ids), PSF_cap, history


    def ReconstructGPU(self, PSF, R_n, mode_ids, A_0=None, A_ref=None, verbous=False, optimize_norm=False):
        C  = []         # optimization criterion
        Hs = []         # interaction matrices for every iteration
        P_MLs = []      # estimators for every iteration
        A_ests = []     # history of estimated coefficients
        
        mode_ids.sort()

        # If coefficients ground truth is known
        if A_ref is not None:
            if np.all(np.abs(self.pack(A_ref)) < 1e-9):
                print('Warning: nothing to optimize, all ref. coefs. = 0')
                return A_ref, None, None

        # Account for the initial approximation for aberrations
        initial_OPD = 0.0
        A_est = cp.zeros(max(mode_ids)+1, dtype=cp.float32)

        if A_0 is not None:
            modes_0 = np.where(np.isfinite(A_0))[0].tolist() # modes defined in the initial vector
            for i in set(mode_ids) & set(modes_0):
                A_est[i] = A_0[i]
            initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, A_0)

        for i in set(range(max(mode_ids)+1)) - set(mode_ids):
            A_est[i] = cp.nan
        A_est = self.pack(A_est)

        self.tel.src.OPD = self.astigmatism_OPD + initial_OPD
        PSF_0 = self.tel.ComputePSF()
    
        # Take a priory object information into account
        if self.tel.object is not None:
            PSF_0 = csg.convolve2d(PSF_0, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        A_ests.append(cp.copy(A_est)) # saving coefficients history
        PSF_cap = cp.array(PSF_0, dtype=cp.float32)      # saving PSF history
        inv_R_n = 1.0 / cp.clip(R_n.reshape(R_n.shape[0]*R_n.shape[1]), a_min=1e-6, a_max=R_n.max())
        
        PSF_buf = cp.array(PSF, dtype=cp.float32) 
        inv_R_n_buf = cp.array(inv_R_n, dtype=cp.float32) 

        criterion  = lambda i: np.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: cp.linalg.norm(v, ord=2)

        norm = 1.0 #normalizes flux

        for i in range(self.iterations):
            if optimize_norm: norm = PSF_cap.max() / PSF.max()

            dI = (PSF_buf*norm-PSF_cap).flatten()
            # Check convergence
            C.append( cp.dot(dI * inv_R_n_buf/norm, dI) )
            
            if i > 0 and (criterion(i) < 1e-6 or coefs_norm(A_ests[i]-A_ests[i-1]) < 1e-12):
                print('Criterion', criterion(i), 'is reached at iter.', i)
                break

            if A_ref is not None:
                if coefs_norm(self.pack(A_ref)-A_est) < 1e-12:
                    print('Err. vec. norm', coefs_norm(self.pack(A_ref)-A_est), 'is reached at iter.', i)
                    break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(coefs=self.unpack(A_est, mode_ids))                               

            # Maximum likelyhood estimation
            P_ML = clg.pinv(H.T * (inv_R_n_buf/norm) @ H) @ H.T * (inv_R_n_buf/norm)
            d_A = P_ML @ dI
            A_est += d_A
            
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(cp.copy(A_est))

            if verbous:
                print( 'Iteration:', i )  
                if A_ref is not None:
                    print( 'Criterion:', criterion(i), ', err. vec. norm:', coefs_norm(self.pack(A_ref)*1e9-A_est*1e9))
                else:
                    print( 'Criterion:', criterion(i))
                self.print_modes(self.unpack(d_A*1e9, mode_ids))
                print()
            
            # Recreate the spot with reconstructed coefficients
            estimated_OPD = self.modeBasis.wavefrontFromModes(self.tel, self.unpack(A_est, mode_ids))
            self.tel.src.OPD = estimated_OPD + self.astigmatism_OPD
            PSF_cap = self.tel.ComputePSF()

            if self.tel.object is not None:
                PSF_cap = csg.convolve2d(PSF_cap, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()

        if optimize_norm:
            PSF_cap = PSF_cap * PSF.max() / PSF_cap.max()

        history = { #save intermediate data every iteration
                'P_ML' : cp.asnumpy( cp.dstack(P_MLs) ),
                'H'    : cp.asnumpy( cp.dstack(Hs) ),
                'A_est': cp.asnumpy( cp.asnumpy(cp.squeeze(cp.dstack(A_ests), axis=0)) ),
                'C'    : cp.asnumpy( cp.array(C) )
            }
        return cp.asnumpy(self.unpack(A_est, mode_ids)), cp.asnumpy(PSF_cap), history


    def Reconstruct(self, PSF, R_n, mode_ids, A_0=None, A_ref=None, verbous=False, optimize_norm=False):
        if self.gpu:
            return self.ReconstructGPU(PSF, R_n, mode_ids, A_0, A_ref, verbous, optimize_norm)
        else:
            return self.ReconstructCPU(PSF, R_n, mode_ids, A_0, A_ref, verbous, optimize_norm)