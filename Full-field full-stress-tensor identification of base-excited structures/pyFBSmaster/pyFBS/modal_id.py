import numpy as np
from scipy import linalg
from .utility import MAC
from PyQt5 import QtCore, QtWidgets
from scipy.optimize import least_squares

class modal_id(object):
    """
    Poly-reference frequency domain identification for modal parameter estimation as a combination of 
    poly-reference Least-Squares Complex Frequency (pLSCF) and Least-Squares Frequency Domain (LSFD) methods.

    :param freq: Frequency range.
    :type ch: array (float)
    :param FRF: Frequency response function matrix in the form of [frequency points, ouputs, inputs].
    :type refch: array (complex)

    References:     
    -----------
    [1] Guillaume, Patrick, et al. "A poly-reference implementation of the least-squares complex 
        frequency-domain estimator." Proceedings of IMAC. Vol. 21. Kissimmee, FL: A Conference & Exposition 
        on Structural Dynamics, Society for Experimental Mechanics, 2003.
        
    """
    
    def __init__(self, freq, FRF):
        self.FRF = FRF
        self.freq = freq
        self.No = FRF.shape[1]
        self.Ni = FRF.shape[2]
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('gui', 'qt')
            
        except BaseException as e:
            # issued if code runs in bare Python
            print('Could not enable IPython gui support: %s.' % e)

        # get QApplication instance
        self.app = QtCore.QCoreApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(['app'])
            self.app.references = set()

    def stabilization(self):  
        from .app_stability import App

        self.win = App(self)

        if hasattr(self.app, 'references'):
            self.app.references.add(self.win)
        else:
            self.app.references = {self.win}
        
    def pLSCF(self, max_order, step_order=2, stab_f=0.01, stab_damp=0.05, stab_mpf=0.05):
        """
        Poly-reference Least-Squares Complex Frequency (pLSCF) method for system's poles and modal
        participation factors estimation.

        :param max_order: Highest order of the polynomial basis functions.
        :type pos: int
        :param step_order: Step between two consecutive model orders.
        :type pos: int, optional
        :param stab_f: variation over consecutive model orders of the natural frequency.
        :type pos: float, optional
        :param stab_damp: variation over consecutive model orders of the damping ratio.
        :type pos: float, optional
        :param stab_mpf: variation over consecutive model orders of the modal participation factor.
        :type pos: float, optional
        
        Note: Treat stabilization of modal participation factors with care for models with low number of
        inputs (consider increasing variation criterion).
        """
        
        stab_plot = np.asarray([[0,1e-12,1e-12,0]])
        p = [[np.array([0])]]
        L = [[np.zeros((self.Ni,1))]]
        
        d = self.freq[-1] - self.freq[0]
        r = np.arange(max_order)

        exp_term_ = np.exp(1.j*np.pi/d*np.einsum('ij,k->kij',r[:,np.newaxis],self.freq))
        exp_term = exp_term_@np.conj(exp_term_).transpose(0,2,1)

        R = np.einsum("ijk->jk",exp_term).real
        S = -np.einsum("ijk, ilm", self.FRF, exp_term).transpose(0,2,3,1).reshape(self.No,(max_order),(max_order)*self.Ni).real
        T = np.einsum("ijk, ilm", np.conj(self.FRF).transpose(0,2,1) @ self.FRF,
                      exp_term).transpose(2,0,3,1).reshape((max_order)*self.Ni,(max_order)*self.Ni).real
        
        M = (T - np.sum(S.transpose(0,2,1) @ np.linalg.pinv(R) @ S, axis=0))

        n_p_all = np.arange(1,max_order+1,step_order)
        for n_p in n_p_all:
            # print progress
            if n_p != n_p_all[-1]:
                print(f"Order:" +str(n_p+1) +'/'+ str(n_p_all[-1]+1), end="\r")
            else:
                print(f"Order:" +str(n_p+1) +'/'+ str(n_p_all[-1]+1), end="\n")

            # alpha coefficients
            alpha = np.block([[np.linalg.solve(-M[:n_p*self.Ni,:n_p*self.Ni],M[:n_p*self.Ni,n_p*self.Ni:(n_p+1)*self.Ni])],
                      [np.eye(self.Ni)]])[:-self.Ni]

            # companion matrix
            A = np.block([[-alpha.reshape(n_p,self.Ni,self.Ni)[::-1].reshape(alpha.shape).T],
                  [np.eye(self.Ni*(n_p-1),self.Ni*n_p)]])

            # poles
            eigval, eigvec = linalg.eig(A)
            eigval_t = -1*np.log(eigval)*2*d

            f_np, damp_np, p_np, L_np = self.transform_poles(eigval_t, eigvec, self.Ni, self.freq) 

            # append results
            if p_np.size == 0:
                stab_plot = np.vstack((stab_plot, [n_p+1,1e-12,1e-12,0]))
                p.append([np.array([0])])
                L.append([np.zeros((self.Ni,1))])
            else:
                p.append([p_np])
                L.append([L_np])
                stab_plot = np.vstack((stab_plot,
                    self.select_stable_poles(stab_plot,L,n_p,f_np,damp_np,stab_f,stab_damp,stab_mpf)))
        
        self.stab_plot = stab_plot[1:]
        self.poles = p[1:]
        self.mpf = L[1:]

    def pLSRA(self, max_order, step_order=2, stab_f=0.01, stab_damp=0.05, stab_mpf=0.05, sol_type = 'iterative'):
        """
        Perform polyreference least squares rational approximation (pLSRA) on frequency response data.

        Args:
            max_order (int): Maximum polynomial order of the rational approximation.
            step_order (int, optional): Step size between consecutive orders. Defaults to 2.
            stab_f (float, optional): Stability criterion for the frequency component. Defaults to 0.01.
            stab_damp (float, optional): Stability criterion for the damping component. Defaults to 0.05.
            stab_mpf (float, optional): Stability criterion for the mode participation factor component. Defaults to 0.05.
            sol_type (str, optional): Type of solver to use ('linearized' or 'iterative'). Defaults to 'iterative'.

        Returns:
            None
        """
        import polyrat
               
        stab_plot = np.asarray([[0,1e-12,1e-12,0]])
        p = [[np.array([0])]]
        L = [[np.zeros((self.Ni,1))]]
        
        n_p_all = np.arange(2, max_order+1, step_order)
        for n_p in n_p_all:            
            if sol_type == 'linearized':
                rat = polyrat.LinearizedRationalApproximation(n_p, n_p)

            elif sol_type == 'iterative':
                rat = polyrat.SKRationalApproximation(n_p, n_p, verbose = False)

            else:
                raise Exception('Unknown solver type.')
            
            rat.fit(2*np.pi*self.freq[:,None], self.FRF)
            
            # poles
            poles = 1.j*rat.poles().ravel()

            # mpf
            numerator = rat.numerator(np.abs(poles)[:,None])
            u,s,vh = np.linalg.svd(numerator, full_matrices = False)
            mpf = vh[:,0,:].T

            f_np, damp_np, p_np, L_np = self.transform_poles(poles, mpf, self.Ni, self.freq) 

            # append results
            if p_np.size == 0:
                stab_plot = np.vstack((stab_plot, [n_p+1,1e-12,1e-12,0]))
                p.append([np.array([0])])
                L.append([np.zeros((self.Ni,1))])
            else:
                p.append([p_np])
                L.append([L_np])
                stab_plot = np.vstack((stab_plot,
                    self.select_stable_poles(stab_plot,L,n_p-1,f_np,damp_np,stab_f,stab_damp,stab_mpf)))
                
                # print progress
            if n_p != n_p_all[-1]:
                print(f"Order:" +str(n_p) +'/'+ str(n_p_all[-1]), end="\r")
            else:
                print(f"Order:" +str(n_p) +'/'+ str(n_p_all[-1]), end="\n")
        
        self.stab_plot = stab_plot[1:]
        self.poles = p[1:]
        self.mpf = L[1:]
    
    def generate_P(self, frf_type, lower_residuals, upper_residuals, freq_rec = None):
        """
        Generate tensor P containing data on denominator and modal participation factors, assuming general viscous damping model.

        Args:
            frf_type (str): Type of FRF ('receptance', 'mobility', or 'accelerance').
            lower_residuals (bool): Flag indicating if lower residuals should be included in P.
            upper_residuals (bool): Flag indicating if upper residuals should be included in P.
            freq_rec (ndarray, optional): Frequency vector for FRF. Defaults to None.

        Returns:
            tuple: A tuple containing the number of selected poles (m) and the generated tensor P.
        """
        
        # prepare input data
        m = self.selected_poles.shape[0]
        
        if m == 0:
            raise Exception("No pole is selected. Select at least one pole on stability chart.")
                
        sr = self.selected_poles[None].real
        si = self.selected_poles[None].imag
        Lr = self.selected_mpf[None].real
        Li = self.selected_mpf[None].imag
        if freq_rec is None:
            w = 2*np.pi*self.freq[:,None,None]
        else:
            # apply a different frequency vector
            w = 2*np.pi*freq_rec.ravel()[:,None,None]
            
        # adjust analytical model for frf_type
        if frf_type == 'receptance':  
            pO1r = -((Li*si + Lr*sr - Li*w)/(sr**2 + (-si + w)**2)) - (Lr*sr + Li*(si + w))/(sr**2 + (si + w)**2)
            pO1i = (Li*sr + Lr*(-si + w))/(sr**2 + (-si + w)**2) + (Li*sr - Lr*(si + w))/(sr**2 + (si + w)**2)

            pO2r = (Lr*si - Li*sr - Lr*w)/(sr**2 + (-si + w)**2) + (Li*sr - Lr*(si + w))/(sr**2 + (si + w)**2)
            pO2i = -((Li*si + Lr*sr - Li*w)/(sr**2 + (-si + w)**2)) + (Lr*sr + Li*(si + w))/(sr**2 + (si + w)**2)

            if lower_residuals:
                pL1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , -1/w**2)        
                pL2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , -1/w**2)      

            if upper_residuals:
                pU1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , np.ones_like(w))        
                pU2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , np.ones_like(w))
                
        elif frf_type == 'mobility':
            pO1r = (Lr*si*w - Li*sr*w + Lr*w**2)/(sr**2 + (si + w)**2) + (Li*sr*w + Lr*(-(si*w) + w**2))/(sr**2 + (-si + w)**2)
            pO1i = (Li*si*w + Lr*sr*w - Li*w**2)/(sr**2 + (-si + w)**2) - (Li*si*w + Lr*sr*w + Li*w**2)/(sr**2 + (si + w)**2)

            pO2r = (-((Li*si + Lr*sr)*w) + Li*w**2)/(sr**2 + (-si + w)**2) - (Li*si*w + Lr*sr*w + Li*w**2)/(sr**2 + (si + w)**2)
            pO2i = (Li*sr*w + Lr*(-(si*w) + w**2))/(sr**2 + (-si + w)**2) + (Li*sr*w - Lr*(si*w + w**2))/(sr**2 + (si + w)**2)

            if lower_residuals:
                pL1 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , 1/w)        
                pL2 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , -1/w)      

            if upper_residuals:
                pU1 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , -w)        
                pU2 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , w)
            
        elif frf_type == 'accelerance':
            pO1r = (Li*si*w**2 + Lr*sr*w**2 - Li*w**3)/(sr**2 + (-si + w)**2) + (Li*si*w**2 + Lr*sr*w**2 + Li*w**3)/(sr**2 + (si + w)**2)
            pO1i = -((-(Lr*si*w**2) + Li*sr*w**2 + Lr*w**3)/(sr**2 + (-si + w)**2)) + (Lr*si*w**2 - Li*sr*w**2 + Lr*w**3)/(sr**2 + (si + w)**2)
            pO2r = (Lr*si*w**2*((-si + w)**2 - (si + w)**2) + Li*sr*w**2*(-(-si + w)**2 + (si + w)**2) + Lr*w**3*(2*sr**2 + (-si + w)**2 + (si + w)**2))/((sr**2 + (-si + w)**2)*(sr**2 + (si + w)**2))
            pO2i = (Li*si*w**2 + Lr*sr*w**2 - Li*w**3)/(sr**2 + (-si + w)**2) - (Li*si*w**2 + Lr*sr*w**2 + Li*w**3)/(sr**2 + (si + w)**2)

            if lower_residuals:
                pL1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , np.ones_like(w))        
                pL2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , np.ones_like(w))      

            if upper_residuals:
                pU1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , -w**2)        
                pU2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , -w**2)
        
        else:
            raise Exception('Unknown parameter frf_type; only "receptance", "mobility" or "accelerance" is allowed.') 
            
        # generate P with regard to the lower and upper residuals
        
        if lower_residuals and not upper_residuals:
            P = np.block([[[pO1r,pO1i,pL1]],[[pO2r,pO2i,pL2]]])
        
        elif upper_residuals and not lower_residuals:
            P = np.block([[[pO1r,pO1i,pU1]],[[pO2r,pO2i,pU2]]])
        
        elif lower_residuals and upper_residuals:
            P = np.block([[[pO1r,pO1i,pL1,pU1]],[[pO2r,pO2i,pL2,pU2]]])
        
        elif not lower_residuals and not upper_residuals:
            P = np.block([[[pO1r,pO1i]],[[pO2r,pO2i]]])
            
        else:
            raise Exception('Error generating P.')
        
        self.assuming_proportional = False
        return m, P
    
    def generate_P_proportional(self, frf_type, lower_residuals, upper_residuals, freq_rec = None):
        """
        Generate tensor P containing data on denominator and (real) modal participation factors, assuming proportional viscous damping model.

        Args:
            frf_type (str): Type of FRF ('receptance', 'mobility', or 'accelerance').
            lower_residuals (bool): Flag indicating if lower residuals should be included in P.
            upper_residuals (bool): Flag indicating if upper residuals should be included in P.
            freq_rec (ndarray, optional): Frequency vector for FRF. Defaults to None.

        Returns:
            tuple: A tuple containing the number of selected poles (m) and the generated tensor P.
        """
        
        # prepare input data
        m = self.selected_poles.shape[0]
        
        if m == 0:
            raise Exception("No pole is selected. Select at least one pole on stability chart.")
                
        wr = np.abs(self.selected_poles)[None]
        xir = (-np.real(self.selected_poles)/np.abs(self.selected_poles))[None]

        self.mpf_real = self.complex_to_closest_real(self.selected_mpf)
        Lr = self.mpf_real[None]
        
        if freq_rec is None:
            w = 2*np.pi*self.freq[:,None,None]
        else:
            # apply a different frequency vector
            w = 2*np.pi*freq_rec.ravel()[:,None,None]
            
        # adjust analytical model for frf_type
        if frf_type == 'receptance':  
            pO1r = (Lr*(-w**2 + wr**2))/((-w**2 + wr**2)**2 + 4*w**2*wr**2*xir**2)

            pO2r = (-2*Lr*w*wr*xir)/((-w**2 + wr**2)**2 + 4*w**2*wr**2*xir**2)

            if lower_residuals:
                pL1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , -1/w**2)        
                pL2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , -1/w**2)      

            if upper_residuals:
                pU1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , np.ones_like(w))        
                pU2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , np.ones_like(w))
                
        elif frf_type == 'mobility':
            pO1r = (2*Lr*w**2*wr*xir)/((-w**2 + wr**2)**2 + 4*w**2*wr**2*xir**2)

            pO2r = (Lr*w*(-w**2 + wr**2))/((-w**2 + wr**2)**2 + 4*w**2*wr**2*xir**2)

            if lower_residuals:
                pL1 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , 1/w)        
                pL2 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , -1/w)      

            if upper_residuals:
                pU1 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , -w)        
                pU2 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , w)
            
        elif frf_type == 'accelerance':
            pO1r = (Lr*(w**4 - w**2*wr**2))/((-w**2 + wr**2)**2 + 4*w**2*wr**2*xir**2)
            
            pO2r = (2*Lr*w**3*wr*xir)/((-w**2 + wr**2)**2 + 4*w**2*wr**2*xir**2)

            if lower_residuals:
                pL1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , np.ones_like(w))        
                pL2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , np.ones_like(w))      

            if upper_residuals:
                pU1 = np.kron(np.kron(np.eye(self.Ni),np.array([1,0])) , -w**2)        
                pU2 = np.kron(np.kron(np.eye(self.Ni),np.array([0,1])) , -w**2)
        
        else:
            raise Exception('Unknown parameter frf_type, please use receptance, mobility or accelerance.')
            
        # generate P with regard to the lower and upper residuals
        
        if lower_residuals and not upper_residuals:
            P = np.block([[[pO1r,pL1]],[[pO2r,pL2]]])
        
        elif upper_residuals and not lower_residuals:
            P = np.block([[[pO1r,pU1]],[[pO2r,pU2]]])
        
        elif lower_residuals and upper_residuals:
            P = np.block([[[pO1r,pL1,pU1]],[[pO2r,pL2,pU2]]])
        
        elif not lower_residuals and not upper_residuals:
            P = np.block([[[pO1r]],[[pO2r]]])
            
        else:
            raise Exception('Error generating P.')

        self.assuming_proportional = True    
        return m, P
    
    def pLSFD(self, frf_type = 'receptance', assume_proportional = False, reconstruction = True, lower_residuals = True, upper_residuals = True, W = None, freq_rec = None, parsing = [None, None]):
        """
        Given the poles and modal participation factors, the remaining modal parameters are estimated with a Least-Squares Frequency Domain (LSFD) method.

        :param frf_type: Define FRF type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str, optional
        :param assume_proportional: Decide if proportional damping model should be considered in the least squares fit.
        :type assume_proportional: bool, optional
        :param reconstruction: Reconstruct FRF from estimated modal parameters.
        :type reconstruction: bool, optional
        :param lower_residuals: Compute lower residuals.
        :type lower_residuals: bool, optional
        :param upper_residuals: Compute upper residuals.
        :type upper_residuals: bool, optional
        :param W: Weighting vector.
        :type W: float, freq_like array of positive (nonzero) values
        :param freq_rec: Perform FRF reconstruction given the estimated modal parameters.
        :type freq_rec: bool

        :returns shape: Non-normalized mode shape.
        :returns FRF_rec: FRF matrix reconstructed from identified modal parameters.
        :returns LR: Lower residual.
        :returns UR: Upper residual.
        :returns residues: Full identified residue matrix.
        """

        # prepare inputs for the least squares solver
        if (np.array(parsing) == None).all():
            Y_ = np.block([[[self.FRF.real]],[[self.FRF.imag]]])
        
            if not assume_proportional: 
                # general viscous damping model
                m, P_ = self.generate_P(frf_type, lower_residuals, upper_residuals, freq_rec = None)
            else:
                # proportional visous damping model
                m, P_ = self.generate_P_proportional(frf_type, lower_residuals, upper_residuals, freq_rec = None)

        else:
            interval_range_1 = int(parsing[0])
            interval_range_2 = int(parsing[1])

            if not interval_range_2 > interval_range_1:
                raise Exception("Interval range 2 should be greater than interval range 1.")

            nat_freq_ind = np.array([np.where(np.isclose(self.freq, _, atol = self.freq[1]-self.freq[0]))[0][0] for _ in self.nat_freq])

            level_1_intervals = nat_freq_ind[:,None] + np.array([-interval_range_1,interval_range_1])
            level_1_ind = np.array([np.arange(*_) for _ in level_1_intervals]).ravel()
            level_1_ind = np.unique(level_1_ind[level_1_ind > 0])
            
            level_2_intervals = nat_freq_ind[:,None] + np.array([-interval_range_2,interval_range_2])
            level_2_ind = np.array([np.arange(*_) for _ in level_2_intervals]).ravel()
            level_2_ind = np.unique(level_2_ind[level_2_ind > 0])
            
            ind_list = np.setdiff1d(level_2_ind, level_1_ind)

            Y_ = np.block([[[self.FRF.real[ind_list]]],[[self.FRF.imag[ind_list]]]])

            if not assume_proportional: 
                # general viscous damping model
                m, P_ = self.generate_P(frf_type, lower_residuals, upper_residuals, freq_rec = self.freq[ind_list])
            else:
                # proportional visous damping model
                m, P_ = self.generate_P_proportional(frf_type, lower_residuals, upper_residuals, freq_rec = self.freq[ind_list])

        # solve the least squares problem
        if W == None:
            O_ = np.linalg.lstsq((P_.transpose(1,0,2).reshape(-1, P_.shape[-1])), \
                                (Y_.transpose(2,0,1).reshape(-1, Y_.shape[-2])))[0]    
        
        else:
            # weighted least squares solution
            if W.shape == self.freq.shape:
                W_ = np.diag(np.tile(np.tile(W, self.Ni), 2))
                O_ = np.linalg.lstsq(W_**0.5 @ (P_.transpose(1,0,2).reshape(-1, P_.shape[-1])), \
                                    W_**0.5 @ (Y_.transpose(2,0,1).reshape(-1, Y_.shape[-2])))[0]
                
            else:
                raise Exception('Invalid W; shapes of W and freq should be the same.')
        
        # postprocess the solution
        if not self.assuming_proportional:
            # general viscous damping model
                # flexible residues - complex
            O = (O_[:m] + 1.j*O_[m:2*m]).T
            self.R = np.einsum("om, im -> moi", O, self.selected_mpf)
            self.shape = O

                # residual residues - complex
            if lower_residuals and upper_residuals:
                RL_, RU_ = np.split(O_[2*m:], [-2*self.Ni])

                self.RL = (RL_[::2] + 1.j*RL_[1::2]).T
                self.RU = (RU_[::2] + 1.j*RU_[1::2]).T

            elif lower_residuals and not upper_residuals:
                RL_ = O_[2*m:]
                self.RL = (RL_[::2] + 1.j*RL_[1::2]).T
                self.RU = np.zeros_like(self.RL)

            elif upper_residuals and not lower_residuals:
                RU_ = O_[2*m:]
                self.RU = (RU_[::2] + 1.j*RU_[1::2]).T
                self.RL = np.zeros_like(self.RU)
                
            elif not upper_residuals and not lower_residuals:
                self.RL = np.zeros((self.No,self.Ni))
                self.RU = np.zeros((self.No,self.Ni))

        else:
            # proportional visous damping model
                # flexible modal constants - real
            O = O_[:m].T
            self.A = np.einsum("om, im -> moi", O, self.mpf_real)
            self.shape = O
            
                # residual modal constants - complex
            if lower_residuals and upper_residuals:
                AL_, AU_ = np.split(O_[m:], [-2*self.Ni])

                self.AL = (AL_[::2] + 1.j*AL_[1::2]).T
                self.AU = (AU_[::2] + 1.j*AU_[1::2]).T

            elif lower_residuals and not upper_residuals:
                AL_ = O_[m:]
                self.AL = (AL_[::2] + 1.j*AL_[1::2]).T
                self.AU = np.zeros_like(self.AL)

            elif upper_residuals and not lower_residuals:
                AU_ = O_[m:]
                self.AU = (AU_[::2] + 1.j*AU_[1::2]).T
                self.AL = np.zeros_like(self.AU)
                
            elif not upper_residuals and not lower_residuals:
                self.AL = np.zeros((self.No,self.Ni))
                self.AU = np.zeros((self.No,self.Ni))

        
        # frf reconstruction
        if reconstruction:
            if freq_rec != None:
                if not self.assuming_proportional: 
                    # general viscous damping model
                    _, P_ = self.generate_P(frf_type, lower_residuals, upper_residuals, freq_rec = freq_rec)
                else:
                    # proportional visous damping model
                    _, P_ = self.generate_P_proportional(frf_type, lower_residuals, upper_residuals, freq_rec = freq_rec)
                Y_rec_ = np.einsum("fip,po->foi", P_ , O_)
            if not (np.array(parsing) == None).all(): # if parsing is not None, than P_ needs to be reevaluated before reconstruction due to previous slicing
                if not self.assuming_proportional: 
                    # general viscous damping model
                    _, P_ = self.generate_P(frf_type, lower_residuals, upper_residuals, freq_rec = None)
                else:
                    # proportional visous damping model
                    _, P_ = self.generate_P_proportional(frf_type, lower_residuals, upper_residuals, freq_rec = None)
                Y_rec_ = np.einsum("fip,po->foi", P_ , O_)
            else:
                Y_rec_ = np.einsum("fip,po->foi", P_ , O_)   
            
            self.FRF_rec = np.vectorize(complex)(*np.split(Y_rec_,2))

    def normalize(self, output_dp_ind, input_dp_ind, check_dp = True):
        """
        If proportional damping is NOT assumed, the function performs a-normalization of estimated flexible residues.
        If proportional damping IS assumed, the function performs mass normalization of flexible modal constants.
        
        :param output_dp_ind: define index of the output driving point in the FRF matrix
        :type output_dp_ind: int list
        :param input_dp_ind: define index of the input driving point in the FRF matrix
        :type input_dp_ind: int list
        :param check_dp: check if driving point values are physically meaningful
        :type check_dp: bool
        
        If proportional damping is NOT assumed:
        :returns Psi_o: A-normalized output mode shapes (complex).
        :type Psi_o: array [no. output DoFs x no. modes]
        :returns Psi_i: A-normalized input mode shapes (complex).
        :type Psi_i: array [no. input DoFs x no. modes]

        If proportional damping IS assumed:
        :returns Phi_o: mass-normalized output mode shapes (real).
        :type Phi_o: array [no. output DoFs x no. modes]
        :returns Phi_i: mass-normalized input mode shapes (real).
        :type Phi_i: array [no. input DoFs x no. modes]
        """
        
        output_dp_ind_ = np.copy(output_dp_ind)
        input_dp_ind_ = np.copy(input_dp_ind)

        # a-normalization of residues
        if not self.assuming_proportional:
            print('A-normalization')
            psi_o = []
            psi_i = []

            for r, R_r in enumerate(self.R):
                dp_values = R_r[output_dp_ind, input_dp_ind]

                if check_dp:
                    # check if driving point values are physically meaningfull
                        # disregard entries with non-negative imaginary values
                    ind_1 = np.where(np.sign(dp_values.imag) == 1)[0]
                    dp_values = np.delete(dp_values, ind_1)
                    output_dp_ind_ = np.delete(output_dp_ind, ind_1)
                    input_dp_ind_ = np.delete(input_dp_ind, ind_1)

                        # check the remaining set for near-zero values (less than 1% average)
                    ind_2 = np.where(np.abs(dp_values) < 0.01*np.mean(np.abs(dp_values)))[0]
                    dp_values = np.delete(dp_values, ind_2)
                    output_dp_ind_ = np.delete(output_dp_ind_, ind_2)
                    input_dp_ind_ = np.delete(input_dp_ind_, ind_2)

                    # check kow many values are left
                    if len(dp_values) > 0:
                        ndp = len(dp_values)
                        print('Mode '+ str(r+1) + ' - Passed: ' + str(ndp) + '/' + str(len(output_dp_ind)))
                    else:
                        raise Exception('For at least one mode none of the provided driving point values seems to be physically valid. If you want to proceed anyway, apply check_dp = False.')

                else:
                    ndp = len(dp_values)

                # least squares solution (of absolute values due to the global/absolute phase shifts between residue vectors)
                    # columns -> modeshapes
                b_output = np.hstack(R_r[:,input_dp_ind_].T) # stacked residue columns at input_dp_dof ind
                a_output = (np.repeat(dp_values**0.5,self.R.shape[1])[:,None] * np.tile(np.eye(self.R.shape[1]),ndp).T)
                x_abs_output = np.linalg.lstsq(a_output, np.abs(b_output))[0]
                psi_o.append(x_abs_output*np.exp(1.j*np.angle(R_r[:,0])))

                # rows -> modal participation factors
                b_input = np.hstack(R_r[output_dp_ind_,:]) # stacked residue rows at output_dp_dof ind
                a_input = (np.repeat(dp_values**0.5,self.R.shape[2])[:,None] * np.tile(np.eye(self.R.shape[2]),ndp).T)
                x_abs_input = np.linalg.lstsq(a_input, np.abs(b_input))[0]
                psi_i.append(x_abs_input*np.exp(1.j*np.angle(R_r[0,:])))

            self.Psi_o = np.array(psi_o).T
            self.Psi_i = np.array(psi_i).T
        
        # mass normalization of modal constants
        elif self.assuming_proportional:
            print('Mass normalization:')
            phi_o = []
            phi_i = []

            for r, A_r in enumerate(self.A):
                dp_values = A_r[output_dp_ind, input_dp_ind]

                if check_dp:
                    # check if driving point values are physically meaningfull
                        # disregard entries with non-negative imaginary values
                    ind_1 = np.where(np.sign(dp_values) == -1)[0]
                    dp_values = np.delete(dp_values, ind_1)
                    output_dp_ind_ = np.delete(output_dp_ind, ind_1)
                    input_dp_ind_ = np.delete(input_dp_ind, ind_1)

                        # check the remaining set for near-zero values (less than 1% average)
                    ind_2 = np.where(np.abs(dp_values) < 0.01*np.mean(np.abs(dp_values)))[0]
                    dp_values = np.delete(dp_values, ind_2)
                    output_dp_ind_ = np.delete(output_dp_ind_, ind_2)
                    input_dp_ind_ = np.delete(input_dp_ind_, ind_2)

                        # check kow many values are left
                    if len(dp_values) > 0:
                        ndp = len(dp_values)
                        print('Mode '+ str(r+1) + ' - Passed: ' + str(ndp) + '/' + str(len(output_dp_ind)))
                    else:
                        raise Exception('For at least one mode none of the provided driving point values seems to be physically valid. If you want to proceed anyway, apply check_dp_values = False.')

                else:
                    ndp = len(dp_values)

                # least squares solution (of absolute values due to the global/absolute phase shifts between residue vectors)
                    # columns -> modeshapes
                b_output = np.hstack(A_r[:,input_dp_ind_].T) # stacked residue columns at input_dp_dof ind
                a_output = (np.repeat(dp_values**0.5,self.A.shape[1])[:,None] * np.tile(np.eye(self.A.shape[1]),ndp).T)
                x_abs_output = np.linalg.lstsq(a_output, np.abs(b_output))[0]
                phi_o.append(x_abs_output*np.sign(A_r[:,0]))

                # rows -> modal participation factors
                b_input = np.hstack(A_r[output_dp_ind_,:]) # stacked residue rows at output_dp_dof ind
                a_input = (np.repeat(dp_values**0.5,self.A.shape[2])[:,None] * np.tile(np.eye(self.A.shape[2]),ndp).T)
                x_abs_input = np.linalg.lstsq(a_input, np.abs(b_input))[0]
                phi_i.append(x_abs_input*np.sign(A_r[0,:]))

            self.Phi_o = np.array(phi_o).T
            self.Phi_i = np.array(phi_i).T
        
    def estimate_Phi_from_Psi(self, complex_to_normal = True):
        """
        Tranforms a normalized complex modeshape (assuming general viscous damping model) to an approximate
        mass normalized modeshape via scaling. mode normalization. Before application check complexity using mode complexity factor (pyFBS.MCF).
        
        :param complex_to_normal: Compute closest real representation of the complex modeshapes.
        :type complex_to_normal: bool
        
        :returns Phi_o_: estimated mass-normalized output mode shapes.
        :type Phi_o_: array [no. output DoFs x no. modes]
        :returns Phi_i_: estimated mass-normalized input mode shapes.
        :type Phi_i_: array [no. input DoFs x no. modes]
        """

        wr = np.abs(self.selected_poles)
        xir = (-np.real(self.selected_poles)/np.abs(self.selected_poles))

        scal_fact = (2j*wr*(1-xir**2)**0.5)**0.5

        if not complex_to_normal:
            self.Phi_o_ = self.Psi_o * scal_fact
            self.Phi_o_ = self.Psi_i * scal_fact
        else:
            self.Phi_o_ = self.complex_to_closest_real(self.Psi_o * scal_fact)
            self.Phi_i_ = self.complex_to_closest_real(self.Psi_i * scal_fact)
    
    @staticmethod
    def transform_poles(poles, mpf, Ni, freq):
        """
        Obtain natural frequencies, damping ratios, corresponding poles and modal participation factors from all poles. 

        :param poles: Position of the channel.
        :param Ni: Number of inputs.
        :param L: Modal participation factors.
        
        :returns f: Natural frequencies.
        :type f: array(float)
        :returns damp: Damping ratios.
        :type damp: array(float)
        """

        f = np.abs(poles)*np.sign(np.imag(poles))/2/np.pi
        damp = -np.real(poles)/np.abs(poles)
        
        # select physically meaningful poles and modal participation factors
        ind = np.array([f > np.min(freq), f < np.max(freq), damp > 0]).all(axis = 0)
        f_pos = f[ind]
        damp_pos = damp[ind]*100
        p = poles[ind]
        L = mpf[-Ni:,ind]

        return f_pos, damp_pos, p, L
    
    @staticmethod
    def complex_to_closest_real(c_all):
        '''
        Compute closest real modeshapes to the given set of complex modeshapes. Before doing so,
        check their complexity using mode complexity factor (pyFBS.MCF).

        :param c_all: set of complex modeshapes
        :type c_all: array of size [no. of DoFs x no. of modes]

        :returns r_all: set of closest real modeshapes
        :type r_all: array of size [no. of DoFs x no. of modes]
        '''
        # real vector (r) = complex vector (c) * e^(i*phi)
        # which translates to: {r,0} = {cr cos[phi]-ci sin[phi]  , cr sin[phi] + ci cos[phi]}
        r_all = []
        for c in c_all.T:
            cr = c.real
            ci = c.imag

            def func(x, cr, ci):
                r = x[:-1] 
                phi = x[-1]
                return np.hstack([r - cr*np.cos(phi) + ci*np.sin(phi), 
                                    cr*np.sin(phi) + ci*np.cos(phi)])

            x0 = np.hstack([np.abs(c) * np.sign(cr), [np.angle(c[0])]])

            lower_bounds = np.hstack([np.full_like(cr, -np.inf), [-np.pi]])
            upper_bounds = np.hstack([np.full_like(cr, np.inf), [np.pi]])

            result = least_squares(func, x0, args=(cr, ci), bounds=(lower_bounds, upper_bounds))
            r_all.append(result.x[:-1])
        return np.array(r_all).T
    
    @staticmethod
    def select_stable_poles(stab_plot, L, order, freq_n1, damp_n1, stab_f, stab_damp, stab_mpf):
        """
        Find stable poles and prepare plotting data for stabilization plot.

        :param stab_plot: Array of natural frequencies and damping ratios for each polynomial order.
        :type pos: array(float)
        :param L: Mode participation factors for each polynomial order.
        :type pos: list
        :param order: Current polynomial order.
        :type pos: int
        :param freq_n1: Natural frequencies at current polynomial order.
        :type pos: array
        :param damp_n1: Damping ratios at current polynomial order.
        :type pos: int
        :param stab_f: variation over consecutive model orders of the natural frequency.
        :type pos: float
        :param stab_damp: variation over consecutive model orders of the damping ratio.
        :type pos: float
        :param stab_mpf: variation over consecutive model orders of the modal participation factor.
        :type pos: float
        
        :returns new_stab_plot: Array of natural frequencies and damping ratios with last model order added. 
        """

        # prepare datasets
        poles_n = stab_plot[np.where(stab_plot[:,0] == np.amax(stab_plot[:,0]))]
        freq_n = poles_n[:,1]
        damp_n = poles_n[:,2]

        # each nat. freq. from n+1 is compared to all nat. freq. from n order
        freq_n = np.transpose([freq_n]*freq_n1.shape[0])
        # calculate relative error between nat. freq. from n+1 and n for all possible combinations, check where error bellow limit
        ind = np.argmin(np.abs(freq_n1 - freq_n)/freq_n, axis=1)
        _ind = (np.abs(freq_n1 - freq_n)/freq_n)[np.arange(freq_n.shape[0]),ind] < stab_f
        ind_n1 = ind[_ind]
        ind_n = np.arange(freq_n.shape[0])[_ind]

        # all poles are new at first
        pole_type = np.zeros(freq_n1.shape[0])
        # indices where frequency is stable, add 1
        pole_type[ind_n1] += 1
        
        # indices where damping is stable, check only combinations from frequency limit, add 1
        _ind = np.abs(damp_n1[ind_n1] - damp_n[ind_n])/damp_n[ind_n] < stab_damp
        ind_damp_n1 = ind_n1[_ind]
        ind_damp_n = ind_n[_ind]
        pole_type[ind_damp_n1] += 1
        
        # modal participation factor check, check only poles stable in freq and damp, add 1
        ind_L = ind_damp_n1[1 - np.diag(MAC(L[-1][0][:,ind_damp_n1],L[-2][0][:,ind_damp_n]).real) < stab_mpf]
        pole_type[ind_L] += 1
        
        # save poles as list: model order / nat_freq / damp_rat / pole_type (3-stable, 2-stable freq&damp, 1-stable freq, 0-unstable)
        new_stab_plot = np.array([np.repeat(order+1, freq_n1.shape[0]), freq_n1, damp_n1, pole_type]).T.tolist()

        return new_stab_plot
    
    def pL_from_index(self, index):
        """
        Select pole and mode participation factor based on selected pole from stabilization plot.

        :param index: Index of selected pole from stabilization plot data.
        :type index: int
        
        :returns p_sel: Selected pole.
        :type complex:
        :returns L_sel: Selected mode participation factor.
        :type complex:
        """
        i_order = self.stab_plot[index,0]
        n_all = np.argwhere(self.stab_plot[:,0] == i_order)
        p_L_i = np.argwhere(np.unique(self.stab_plot[:,0]) == i_order).ravel()[0]
        n_i = np.argwhere(n_all.flatten() == index).ravel()[0]

        p_sel = self.poles[p_L_i][0][n_i]
        L_sel = self.mpf[p_L_i][0][:,n_i]

        return p_sel, L_sel
