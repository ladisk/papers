from .utility import *
from tqdm import tqdm
import scipy as sp
from scipy import linalg

DEFAULT_COLUMNS = ["Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2", "Direction_3"]

def find_locations_in_data_frames(df_1, df_2, additional_columns=[]):
    """Find matching locations of data frames ``df_1`` and ``df_2``.

    :param df_1: Data frame 1
    :type df_1: pandas.DataFrame
    :param df_2: Data frame 2
    :type df_2: pandas.DataFrame
    :return: Vector of matching locations of both data frames.
    :rtype: array(float)
    """
    
    columns = DEFAULT_COLUMNS + additional_columns
    df_1_val = np.array(df_1[columns].values, dtype= np.float64)
    df_2_val = np.array(df_2[columns].values, dtype= np.float64)

    # to prevent numerical errors
    df_1_val = np.round(df_1_val, 6) 
    df_2_val = np.round(df_2_val, 6)

    return np.array(np.all((df_1_val[:, None, :] == df_2_val[None, :, :]), axis=-1).nonzero()).T


def SEMM(Y_num, Y_exp, df_chn_num, df_imp_num, df_chn_exp, df_imp_exp, SEMM_type='fully-extend', red_comp=0, red_eq=0, additional_columns = []):
    """
    This function performs SEMM. It couples numerical (``Y_num``) and experimental (``Y_exp``) model to hybrid model. 

    :param Y_num: Numerical response matrix
    :type Y_num: array(float)
    :param Y_exp: Experimental response matrix
    :type Y_exp: array(float)
    :param df_chn_num: Locations and directions of response in the ``Y_num``
    :type df_chn_num: pandas.DataFrame
    :param df_imp_num: Locations and directions of excitation in the ``Y_num``
    :type df_imp_num: pandas.DataFrame
    :param df_chn_exp: Locations and directions of response in the ``Y_exp``
    :type df_chn_exp: pandas.DataFrame
    :param df_imp_exp: Locations and directions of excitation in the ``Y_exp``
    :type df_imp_exp: pandas.DataFrame
    :param SEMM_type: Defined which type of SEMM will be performed - basic ("basic") or fully extended ("fully-extend") or fully extended with SVD truncation on compatibility or equilibrium ("fully-extend-svd")
    :type SEMM_type: str("basic" or "fully-extend" or "fully-extend-svd")
    :param red_comp: Defines how many maximum singular values will not be taken into account in ensuring compatibility conditions
    :type red_comp: int
    :param red_eq: Defines how many maximum singular values will not be taken into account in ensuring equilibrium conditions
    :type red_eq: int
    :param additional_columns: Aditional columns to check for maching between numerical and experimental model in defined data frames: ``df_chn_num``, ``df_imp_num``, ``df_chn_exp``, ``df_imp_exp``
    :type additional_columns: list
    :return: Hybrid model based on numerical and experimental data
    :rtype: array(float)

    The form of the FRFs in the numerical matrix must match the ``df_chn_num`` and ``df_imp_num`` parameters. 
    The ``df_chn_num`` parameter represents the rows (responses) of the numeric matrix, and the ``df_imp_num`` parameter represents the columns (excitaions) that are presented in the numerical model.
    The same guidelines must also be followed for the experimental model, the corresponding response locationsare defined in the parameter ``df_chn_exp`` and the excitation locations in the parameter ``df_imp_exp``.

    The location and direction of an individual response point in the experimental model must coincide exactly with one location and direction of the response and in the numerical model. 
    The same must be true also for the location and direction of excitation.
    """

    # Validation of input data
    Y_num = np.asarray(Y_num)
    if len(Y_num.shape) != 3:
        raise Exception('Wrong shape of input numerical receptance matrix.')

    if len(Y_exp.shape) != 3:
        raise Exception('Input experimental matrx must be 3D matrix.')

    if df_chn_exp.shape[0] != Y_exp.shape[1]:
        raise Exception('The input channel data frame must contain those DoFs that are represented in the experimental model.')

    if df_imp_exp.shape[0] != Y_exp.shape[2]:
        raise Exception('The input impact data frame must contain those DoFs that are represented in the experimental model.')

    if df_chn_num.shape[0] != Y_num.shape[1]:
        raise Exception('The input channel data frame must contain those DoFs that are represented in the numerical model.')

    if df_imp_num.shape[0] != Y_num.shape[2]:
        raise Exception('The input impact data frame must contain those DoFs that are represented in the numerical model.')

    # Initialization data
    Y_num = np.asarray(np.copy(Y_num)).astype(complex)
    Y_exp = np.asarray(np.copy(Y_exp)).astype(complex)

    # Data preparation for building parent, remowed and overlay model
    # Reviewing all experimental obtained DoFs
    maching_locations_chn = find_locations_in_data_frames(df_chn_num, df_chn_exp, additional_columns)
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception('Not all locations in the channel data frame have their exact locations in the numeric channel data frame.')

    maching_locations_imp = find_locations_in_data_frames(df_imp_num, df_imp_exp, additional_columns)
    if maching_locations_imp.shape[0] != df_imp_exp.shape[0]:
        raise Exception('Not all locations in the impact data frame have their exact locations in the numeric impact data frame.')

    chn_b_dof_ind_num = np.copy(maching_locations_chn[:,0])
    chn_b_dof_ind_exp = np.copy(maching_locations_chn[:,1])
    chn_i_dof_ind_num = np.setdiff1d(np.arange(Y_num.shape[1]),chn_b_dof_ind_num)

    chn_n_b = chn_b_dof_ind_num.shape[0]

    imp_b_dof_ind_num = np.copy(maching_locations_imp[:,0])
    imp_b_dof_ind_exp = np.copy(maching_locations_imp[:,1])
    imp_i_dof_ind_num = np.setdiff1d(np.arange(Y_num.shape[2]),imp_b_dof_ind_num)

    imp_n_b = imp_b_dof_ind_num.shape[0]

    # Parent model    
    Y_par = Y_num[:, np.hstack([chn_i_dof_ind_num,chn_b_dof_ind_num])[:,np.newaxis], np.hstack([imp_i_dof_ind_num,imp_b_dof_ind_num])]
    # Removed model    
    Y_rem = Y_num[:, chn_b_dof_ind_num[:,np.newaxis], imp_b_dof_ind_num] 
    # Overlay model    
    Y_ov = Y_exp[:, chn_b_dof_ind_exp[:,np.newaxis], imp_b_dof_ind_exp]

    if SEMM_type == "basic":
        # Single-line method SEMM - basic form - eq(21)
        Y_SEMM = Y_par-Y_par[:, :, -imp_n_b:] @ np.linalg.inv(Y_rem)@(Y_rem-Y_ov)@np.linalg.pinv(Y_rem)@Y_par[:, -chn_n_b:, :]

    elif SEMM_type == "fully-extend" or "extended":
        # Single-line method SEMM - fully-extend form - eq(31)
        Y_SEMM = Y_par-Y_par@np.linalg.pinv(Y_par[:, -chn_n_b:, :])@(Y_rem-Y_ov)@np.linalg.pinv(Y_par[:, :, -imp_n_b:])@Y_par

    elif SEMM_type == "fully-extend-svd":
        Y_SEMM = Y_par-Y_par@np.linalg.pinv(TSVD(Y_par[:, -chn_n_b:, :], reduction=red_comp))@(Y_rem-Y_ov)@np.linalg.pinv(TSVD(Y_par[:, :, -imp_n_b:], reduction=red_eq))@Y_par

    # reordering
    chn_ind = np.argsort(np.hstack([chn_i_dof_ind_num,chn_b_dof_ind_num]))
    imp_ind = np.argsort(np.hstack([imp_i_dof_ind_num,imp_b_dof_ind_num]))
    
    return Y_SEMM[:,chn_ind[:,np.newaxis],imp_ind]

def identification_algorithm(Y_num, Y_exp, df_num_chn, df_num_imp, df_exp_chn, df_exp_imp, axis = 1,  SEMM_type='fully-extend', red_comp=0, red_eq=0, additional_columns = []):
    """
    This function computes the coherence criterion for the identification of inconsistent measurements.
    The algorithm is based on the SEMM method.

    :param Y_num: Numerical response matrix
    :type Y_num: array(float)
    :param Y_exp: Experimental response matrix
    :type Y_exp: array(float)
    :param df_chn_num: Locations and directions of response in the ``Y_num``
    :type df_chn_num: pandas.DataFrame
    :param df_imp_num: Locations and directions of excitation in the ``Y_num``
    :type df_imp_num: pandas.DataFrame
    :param df_chn_exp: Locations and directions of response in the ``Y_exp``
    :type df_chn_exp: pandas.DataFrame
    :param df_imp_exp: Locations and directions of excitation in the ``Y_exp``
    :type df_imp_exp: pandas.DataFrame
    :param axis: Axis of eliminating measurements, 0 or 1 
    :type axis: int
    :param SEMM_type: Defined which type of SEMM will be performed - basic ("basic") or fully extended ("fully-extend") or fully extended with SVD truncation on compatibility or equilibrium ("fully-extend-svd")
    :type SEMM_type: str("basic" or "fully-extend" or "fully-extend-svd")
    :param red_comp: Defines how many maximum singular values will not be taken into account in ensuring compatibility conditions
    :type red_comp: int
    :param red_eq: Defines how many maximum singular values will not be taken into account in ensuring equilibrium conditions
    :type red_eq: int
    :param additional_columns: Aditional columns to check for maching between numerical and experimental model in defined data frames: ``df_chn_num``, ``df_imp_num``, ``df_chn_exp``, ``df_imp_exp``
    :type additional_columns: list
    :return: Hybrid model based on numerical and experimental data
    :rtype: array(float)
    """
    all_exp_chn = np.arange(df_exp_chn.shape[0])
    all_exp_imp = np.arange(df_exp_imp.shape[0])
    
    sel_freq = np.arange(0, np.min([Y_num.shape[0], Y_exp.shape[0]]), 1)
    
    rconstructd_FRF = np.zeros_like(Y_exp, dtype = 'complex')
    
    if axis == 0:
        for i in tqdm(all_exp_chn):
            sel_chn = np.delete(all_exp_chn, i, axis=0)
            sel_imp = all_exp_imp
            
            chn_index = find_locations_in_data_frames(df_num_chn, df_exp_chn.iloc[[i]])
            analsyed_chn = chn_index[:, 0]
            recast_chn = chn_index[:, 1]
            imp_index = find_locations_in_data_frames(df_num_imp, df_exp_imp.iloc[sel_imp])
            analsyed_imp = imp_index[:, 0]
            recast_imp = imp_index[:, 1]

            rconstructd_FRF[:, i, recast_imp] = SEMM(Y_num, Y_exp[np.ix_(sel_freq, sel_chn, sel_imp)], df_num_chn, df_num_imp, df_exp_chn.iloc[sel_chn], df_exp_imp.iloc[sel_imp], SEMM_type, red_comp, red_eq, additional_columns)[:, analsyed_chn, analsyed_imp]
    elif axis == 1:
        for i in tqdm(all_exp_imp):
            sel_chn = all_exp_chn
            sel_imp = np.delete(all_exp_imp, i, axis=0)
            
            chn_index = find_locations_in_data_frames(df_num_chn, df_exp_chn.iloc[sel_chn])
            analsyed_chn = chn_index[:, 0]
            recast_chn = chn_index[:, 1]
            imp_index = find_locations_in_data_frames(df_num_imp, df_exp_imp.iloc[[i]])
            analsyed_imp = imp_index[:, 0]
            recast_imp = imp_index[:, 1]
            
            rconstructd_FRF[:, recast_chn, i] = SEMM(Y_num[sel_freq, :, :], Y_exp[np.ix_(sel_freq, sel_chn, sel_imp)], df_num_chn, df_num_imp, df_exp_chn.iloc[sel_chn], df_exp_imp.iloc[sel_imp], SEMM_type, red_comp, red_eq, additional_columns)[:, analsyed_chn, analsyed_imp]
            
    coh = coh_frf(Y_exp, rconstructd_FRF, return_average=False)
    return rconstructd_FRF, coh

def SEREP(eig_vec_num, eig_vec_exp, df_chn_num, df_chn_exp):
    """
    This function performs SEREP - and expanssion method in modal domain. 

    :param eig_vec_num: Eigenvectors of the numerical model.
    :type eig_vec_num: array(float)
    :param eig_vec_exp: Eigenvectors of the experimental model.
    :type eig_vec_exp: array(float)
    :param df_chn_num: Response locations and directions for the numerical model.
    :type df_chn_num: pandas.DataFrame
    :param df_chn_exp: Response locations and directions for the experimental model.
    :type df_chn_exp: pandas.DataFrame
    
    :return: hybrid eigenvectors
    :rtype: array(float)
    """

    # Initialization data 
    eig_vec_num = np.asarray(np.copy(eig_vec_num)).astype(complex)
    eig_vec_exp = np.asarray(np.copy(eig_vec_exp)).astype(complex)

    # Input data validation
    if df_chn_num.shape[0] != eig_vec_num.shape[0]:
        raise Exception('Numerical model - Incompatible channel data and eigenvector shape.')
    if df_chn_exp.shape[0] != eig_vec_exp.shape[0]:
        raise Exception('Experimental model - Incompatible channel data and eigenvector shape.')

    # Internal and boundary DoF partition
    maching_locations_chn = find_locations_in_data_frames(df_chn_num, df_chn_exp)
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception('Not all experimental channel locations have a matching location in the numerical channel dataframe.')

    b_dof_ind_num = np.copy(maching_locations_chn[:,0])
    b_dof_ind_exp = np.copy(maching_locations_chn[:,1])
    i_dof_ind_num = np.setdiff1d(np.arange(eig_vec_num.shape[0]),b_dof_ind_num)

    # T matrix generation    
    psi_num_ir = np.copy(eig_vec_num)[i_dof_ind_num,:]
    psi_num_br = np.copy(eig_vec_num)[b_dof_ind_num,:]
    psi_exp_br = np.copy(eig_vec_exp)[b_dof_ind_exp,:]

    if psi_num_br.shape[0] < psi_num_br.shape[1]: print('Underdetermined: n_b < m_par')

    print('Condition number:'+"%.2f" % np.linalg.cond(psi_num_br))
    # T = np.vstack([psi_num_ir @ np.linalg.pinv(psi_num_br), np.eye(psi_num_br.shape[0])])
    T = np.vstack([psi_num_ir @ np.linalg.pinv(psi_num_br), psi_num_br @ np.linalg.pinv(psi_num_br)])
    eig_vec_serep = T @ psi_exp_br

    # Reordering - input dof
    reord_ind = np.argsort(np.hstack([i_dof_ind_num,b_dof_ind_num]))

    return eig_vec_serep[reord_ind,:]

def M_SEMM(eig_val_num, damping_num, eig_vec_num, eig_val_exp, damping_exp, eig_vec_exp, df_chn_num, df_chn_exp, SEMM_type = 'extended', tsvd_rcond = 0, ns_rcond = 1.e-12):
    """
    This function performs M-SEMM, System Equivalent Model Mixing in modal domain. 
    Hybrid model is generated from a numerical and an experimental models using a substructuring approach. 

    :param eig_val_num: Eigenvalues of the numerical model.
    :type eig_val_num: array(float)
    :param damping_num: Modal damping ratios of the numerical model.
    :type damping_num: array(float)
    :param eig_vec_num: Mass normalized eigenvectors of the numerical model.
    :type eig_vec_num: array(float)
    
    :param eig_val_exp: Eigenvalues of the experimental model.
    :type eig_val_exp: array(float)
    :param damping_exp: Modal damping ratios of the experimental model.
    :type damping_exp: array(float)
    :param eig_vec_exp: Mass normalized eigenvectors of the experimental model.
    :type eig_vec_exp: array(float)
    
    :param df_chn_num: Response locations and directions for the numerical model.
    :type df_chn_num: pandas.DataFrame
    :param df_chn_exp: Response locations and directions for the experimental model.
    :type df_chn_exp: pandas.DataFrame

    :param SEMM_type: Selection of the applied decoupling constraints (compatibility and equilibrium).
    :type SEMM_type: str('basic', 'extended')
    :param tsvd_rcond: Relative condition number for small singular value trucation tp weaken the extended decoupling constraints.
    :type tsvd_rcond: float
    :param ns_rcond: Relative condition number for the null-space hybrid eigenvector evaluation.
    :type ns_rcond: float

    :return: hybrid modal parameters (eigenvalues, damping ratios, eigenvectors)
    :rtype: array(float)
    """

    # Initialization data
    eig_val_num = np.asarray(np.copy(eig_val_num)).ravel().astype(float)
    damping_num = np.asarray(np.copy(damping_num)).ravel().astype(float)
    eig_val_exp = np.asarray(np.copy(eig_val_exp)).ravel().astype(float)
    damping_exp = np.asarray(np.copy(damping_exp)).ravel().astype(float)
    
    eig_vec_num = np.asarray(np.copy(eig_vec_num)).astype(float)
    eig_vec_exp = np.asarray(np.copy(eig_vec_exp)).astype(float)

    # Input data validation
    if df_chn_num.shape[0] != eig_vec_num.shape[0]:
        raise Exception('Numerical model - Incompatible channel data and eigenvector shape.')
    if df_chn_exp.shape[0] != eig_vec_exp.shape[0]:
        raise Exception('Experimental model - Incompatible channel data and eigenvector shape.')
    if not eig_val_num.shape[0] == damping_num.shape[0] == eig_vec_num.shape[1]:
        raise Exception('Numerical model - Number of modes at the given modal parameters does not match.')
    if not eig_val_exp.shape[0] == damping_exp.shape[0] == eig_vec_exp.shape[1]:
        raise Exception('Numerical model - Number of modes at the given modal parameters does not match.')

    # Internal and boundary DoF partition
    maching_locations_chn = find_locations_in_data_frames(df_chn_num, df_chn_exp)
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception('Not all experimental channel locations have a matching location in the numerical channel dataframe.')

    b_dof_ind_num = np.copy(maching_locations_chn[:,0])
    b_dof_ind_exp = np.copy(maching_locations_chn[:,1])
    i_dof_ind_num = np.setdiff1d(np.arange(eig_vec_num.shape[0]),b_dof_ind_num)

    n_i = i_dof_ind_num.shape[0]
    n_b = b_dof_ind_num.shape[0]
    n_g = n_i + n_b

    # Parent and overlay model generation 
    # (removed model not needed due to duplicity with the parent model)
    eig_val_par = np.copy(eig_val_num)
    xi_par = np.copy(damping_num)
    eig_vec_par = np.copy(eig_vec_num)[np.hstack([i_dof_ind_num,b_dof_ind_num]),:]

    eig_val_ov = np.copy(eig_val_exp)
    xi_ov = np.copy(damping_exp)
    eig_vec_ov = np.copy(eig_vec_exp)[b_dof_ind_exp,:]

    m_par = eig_val_par.shape[0]     
    m_ov = eig_val_ov.shape[0]

    # (Uncoupled) modal matrices - assume proportional viscous damping
    M_m_par = np.eye(m_par)
    C_m_par = np.diag(2*xi_par*eig_val_par**0.5)
    K_m_par = np.diag(eig_val_par)

    M_m_ov = np.eye(m_ov)
    C_m_ov = np.diag(2*xi_ov*eig_val_ov**0.5)
    K_m_ov = np.diag(eig_val_ov)

    if n_b >= m_par:
        print('Redirected to SEREP due to the mathematical equivalence.')
        eig_vec_serep = SEREP(eig_vec_num, eig_vec_exp, df_chn_num, df_chn_exp)
        return eig_val_ov.real, xi_ov.real, eig_vec_serep.real
        
    else:
        # Number of spurious modes
        n_s = m_par - n_b 
        
        # Coupling step
        P_1 = eig_vec_par[-n_b:]
        P_2 = eig_vec_ov

        P_12 = np.linalg.pinv(P_1) @ P_2
        N_1 = sp.linalg.null_space(P_1)
        B_c = np.block([np.zeros([n_b,m_par]) , eig_vec_par[-n_b:] , -eig_vec_ov])

        # Decoupling step
        def generate_N3(eigval_ov):
            if SEMM_type == 'basic':
                P3 = np.copy(P_1)
                N3 = np.copy(N_1)

            elif SEMM_type == 'extended':
                denom = eig_val_par - eigval_ov

                if np.isclose(denom,0).any():
                    u = eig_vec_par[:,np.where(np.isclose(denom,0))[0]]
                    W = u.T
                else:
                    inp = np.einsum('kij,k->ij',np.einsum("ik,jk->kij",eig_vec_par,eig_vec_par[-n_b:]),1/denom)
                    u = sp.linalg.orth(inp, tsvd_rcond)
                    W = u.T

                P3 = W @ eig_vec_par
                N3 = sp.linalg.null_space(P3)

            return N3
        
        def generate_Dsemm(Dov,Dpar,N3):
            return np.block([[Dov,                 np.zeros([m_ov,n_s]),                 P_12.T @ Dpar @ N3],
                            [np.zeros([n_s,m_ov]), np.diag(np.random.random(n_s)*1e-20), N_1.T @ Dpar @ N3],
                            [N3.T @ Dpar @ P_12,   N3.T @ Dpar @ N_1,                    N3.T @ Dpar @ N3]])
        
        def generate_Lpar(N3):
            return np.block([P_12, N_1, N3])
    
        # Output data initialization
        eig_val_semm = np.empty(0)
        xi_semm = np.empty(0)
        eig_vec_semm = np.empty((n_g,0))

        # Copy just to be on the safe side
        _eig_val_ov = np.copy(eig_val_ov)
        _eig_vec_ov = np.copy(eig_vec_ov)

        # Solve for hybrid modal parameters
        for k, (lam_k, phi_ov_k) in enumerate(zip(_eig_val_ov, _eig_vec_ov.T)):
            # One evaluation for 'basic' and "frequency" dependent evaluations for 'extended' formulation 
            if SEMM_type == 'basic' and k == 0 or SEMM_type == 'extended':
                N_3 = generate_N3(lam_k)
                L_par = generate_Lpar(N_3)            

                M_semm = generate_Dsemm(M_m_ov, M_m_par, N_3)
                C_semm = generate_Dsemm(C_m_ov, C_m_par, N_3)
                K_semm = generate_Dsemm(K_m_ov, K_m_par, N_3)

            try:
                _phi_k = sp.linalg.null_space(K_semm - lam_k * M_semm, rcond = ns_rcond)
                np.argmax(_phi_k)
            except:
                # Just in case the null-space evaluation fails
                print('Eigenvalue solver employed.')
                lam_k, _phi_k = sp.sparse.linalg.eigs(K_semm, k=1, M=M_semm, sigma = lam_k)
            
            # mass normalization              
            _phi_k /= np.diagonal(_phi_k.T @ M_semm @ _phi_k)**0.5
            phi_k = eig_vec_par @ L_par @ _phi_k

            if phi_k.shape[1] > 1:
                #print('Spurious solution detected for mode ' + str(k))
                true_ind = [np.argmax(MAC(phi_k[-n_b:],phi_ov_k, output_type = 'diagonal').real)]
                phi_k = phi_k[:,true_ind]
       
            xi_k = np.diagonal(_phi_k.T @ C_semm @ _phi_k) / 2 / lam_k**0.5
             
            eig_val_semm = np.append(eig_val_semm, lam_k.ravel())
            xi_semm = np.append(xi_semm, xi_k.ravel())
            eig_vec_semm = np.append(eig_vec_semm, phi_k, axis = 1)

        # Reordering to input dof order
        reord_ind = np.argsort(np.hstack([i_dof_ind_num,b_dof_ind_num]))
        eig_vec_semm = eig_vec_semm[reord_ind,:]

        return eig_val_semm.real, xi_semm.real, eig_vec_semm.real