import numpy as np
from scipy.linalg import block_diag, norm
from .utility import coh_frf,CMIF
from .VPT import VPT

class SVT(object):
    """
    Singular Vector Transformation - enables transformation of measured responses to a singular DoFs. The left and right
    singular vectors are used as a reduction basis. Care should be taken that the assumption of collocation is taken
    into account.

    :param ch: A DataFrame containing information on channels (i.e. outputs)
    :type ch: pd.DataFrame
    :param refch: A DataFrame containing information on reference channels (i.e. inputs)
    :type refch: pd.DataFrame
    :param freq: Frequency vector
    :type freq: array(float)
    :param FRF: A matrix of Frequency Response Functions FRFs [f,out,in].
    :type FRF: array(float)
    :param grouping_no: Grouping number where SVT is defined.
    :type grouping_no: int
    :param no_svs: Number of singular DoFs.
    :type no_svs: int
    :param Wu: Displacement weigting matrix
    :type Wu: array(float), optional
    :param Wf: Force weighting matrix
    :type Wf: array(float), optional
    """

    def __init__(self, ch, refch, freq,FRF,grouping_no,no_svs,Wu = None, Wf = None):
        # Load the physical input-output DoFs
        self.Channels = ch
        self.RefChannels = refch
        self.Group_No = grouping_no
        self.no_svs = no_svs

        # Load the FRF matrix
        self.freq = freq
        self.FRF = FRF

        # Define the IDM_U and IDM_F matrix
        ind_chn = VPT.find_group(self.Group_No, self.Channels.Grouping.to_numpy())
        ind_imp = VPT.find_group(self.Group_No, self.RefChannels.Grouping.to_numpy())

        # Define a FRF subset
        sub_FRF = self.FRF[:, ind_chn, :][:, :, ind_imp]

        # Calculate left and right singular vectors with
        self.U, self.S, self.V = CMIF(sub_FRF, return_svector=True)

        # Without weighting matrixes
        #self.Tu = np.transpose(np.conj(U[:, :, :self.no_svs]), (0, 2, 1))
        #self.Tf = V[:, :, :self.no_svs]


        # Define reduction matrixes
        Ru = self.U[:, :, :self.no_svs]
        if Wu == None:
            Wu = np.identity(len(ind_chn))

        Rf = self.V[:, :, :self.no_svs]
        if Wf == None:
            Wf = np.identity(len(ind_imp))

        self.Ru = Ru
        self.Rf = Rf

        # SV transormation matrixes
        self.Tu = np.linalg.pinv(self.H(Ru) @ Wu @ Ru) @ self.H(Ru) @ Wu
        self.Tf = Wf @ Rf @ np.linalg.pinv(self.H(Rf) @ Wf @ Rf)

        # Back projection
        self.Fu = Ru @ self.Tu
        self.Ff = Rf @ self.H(self.Tf)

    def H(self,FRF):
        """
        Performs a Hermition transpose on an FRF matrix.

        :param R: A FRF matrix.
        :type R: array(float)
        :return: Hermitian transpose of an FRF matrix
        """
        return np.conj(np.transpose(FRF, (0, 2, 1)))

    def apply_SVT(self,ch, refch, freq, FRF):
        """
        Applies the singular vector transformation on an FRF matrix

        :param ch: A DataFrame containing information on channels (i.e. outputs)
        :type ch: pd.DataFrame
        :param refch: A DataFrame containing information on reference channels (i.e. inputs)
        :type refch: pd.DataFrame
        :param freq: Frequency vector
        :type freq: array(float)
        :param FRF: A matrix of Frequency Response Functions FRFs [f,out,in].
        :type FRF: array(float)
        """

        # Find an overlap between
        ind_chn = VPT.find_group(self.Group_No, ch.Grouping.to_numpy())
        ind_imp = VPT.find_group(self.Group_No, refch.Grouping.to_numpy())

        # Find non-overlapping section
        overlap_chn = np.setxor1d(ind_chn, range(ch.shape[0]))
        overlap_imp = np.setxor1d(ind_imp, range(refch.shape[0]))

        # Define global transformation matrixes
        TU_g = np.zeros((len(freq), self.no_svs + len(overlap_chn), ch.shape[0]), dtype=complex)
        TF_g = np.zeros((len(freq), refch.shape[0], self.no_svs + len(overlap_imp)), dtype=complex)

        # Insert data in global matrixes
        for i,_ind in enumerate(ind_chn):
            TU_g[:,:self.no_svs,_ind] = self.Tu[:,:,i]

        for i,_ind in enumerate(ind_imp):
            TF_g[:,_ind,:self.no_svs] = self.Tf[:,i,:]

        # not-transformed channels append
        i = np.identity(len(overlap_chn))
        I_chn = np.transpose(np.dstack([i] * len(freq)), (2, 0, 1))

        i = np.identity(len(overlap_imp))
        I_imp = np.transpose(np.dstack([i] * len(freq)), (2, 0, 1))

        for i,_ind in enumerate(overlap_chn):
            TU_g[:,self.no_svs:,_ind] = I_chn[:,:,i]

        for i,_ind in enumerate(overlap_imp):
            TF_g[:,_ind,self.no_svs:] = I_imp[:,:,i]

        # Apply the SVT on FRF matrix
        tran_FRF = TU_g @ FRF @ TF_g

        return TU_g,TF_g,tran_FRF

    def consistency(self, grouping,FRF):
        """
        Evaluates SVT consistency indicators based on the supplied grouping number.

        :param grouping: Grouping number
        :type grouping: int
        :param FRF: A matrix of Frequency Response Functions FRFs [f,out,in].
        :type FRF: array(float)
        """

        # get all groupings from the vpt
        _ch_all = self.Channels.Grouping.to_numpy()
        _Rch_all = self.RefChannels.Grouping.to_numpy()

        # extract the grouping mask
        ind_ch = VPT.find_group(grouping, _ch_all)
        ind_Rch = VPT.find_group(grouping, _Rch_all)

        sub_Y = np.transpose(FRF, (1, 2, 0))[ind_ch, :, :][:, ind_Rch, :]
        sub_Fu = self.Fu

        u_f = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)
        u = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            u_f[:, :, i] = sub_Fu[i,:,:] @ sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))
            # initial response
            u[:, :, i] = sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))

        self.u_f = u_f[:, 0, :]
        self.u = u[:, 0, :]

        # Calculate overall sensor consistency indicator
        self.overall_sensor = norm(self.u_f, axis=0) / norm(self.u, axis=0)

        # Calculate specific sensor consistency indicator
        specific_sensor = []
        for i in range(self.u.shape[0]):
            specific_sensor.append(coh_frf(self.u_f[i, :], self.u[i, :]))

        self.specific_sensor = np.asarray(specific_sensor)

        # Calculate impact consistency
        sub_Y = np.transpose(FRF,(1,2,0))[:, ind_Rch, :]
        sub_Ff = self.Ff

        y_f = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)
        y = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            y_f[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i] @ sub_Ff[i,:,:]).T
            # initial response
            y[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i]).T

        self.y_f = y_f[:,0,:]
        self.y = y[:,0,:]

        # Calculate overall impact consistency indicator
        self.overall_impact = norm(self.y_f,axis = 0) / norm(self.y,axis = 0)

        # Calculate specific impact consistency indicator
        specific_impact = []
        for i in range(self.y.shape[0]):
            specific_impact.append(coh_frf(self.y_f[i,:],self.y[i,:]))

        self.specific_impact = np.asarray(specific_impact)

