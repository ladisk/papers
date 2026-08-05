import numpy as np
import pandas as pd
from scipy.linalg import block_diag, norm
from .utility import coh_frf

class VPT(object):
    """
    Virtual Point Transformation (VPT) - enables the transformation of measured responses and loads to virtual DoFs. 
    Current implementation enables the use of rigid and simple flexible interface deformation modes. DoFs supported 
    are 3 translations + 3 rotations + 3 extensions + 3 torsions + 6 skewing. DoFs can be arbitrarily selected. 
    
    The following DoF labels should be used in VP dataframes to include them in the transformation:
    * Translational response/load:
        ux, uy, uz / fx, fy, fz
    * Rotational response/load:
        rx, ry, rz / mx, my, mz
    * Extensional response/load:
        ex, ey, ez / ex, ey, ez
    * Torsional response/load
        tx, ty, tz / tx, ty, tz
    * Skewing response/load:
        sxy, sxz, syz, syx, szx, szy / sxy, sxz, syz, syx, szx, szy

    :param ch: A DataFrame containing information on channels (i.e. outputs)
    :type ch: pd.DataFrame
    :param refch: A DataFrame containing information on reference channels (i.e. inputs)
    :type refch: pd.DataFrame
    :param vp_ch: A DataFrame containing information on virtual point channels
    :type vp_ch: pd.DataFrame
    :param vp_refch: A DataFrame containing information on virtual point loads
    :type vp_refch: pd.DataFrame
    :param Wu: Displacement weigting matrix for the interface channels
    :type Wu: 2D matrix (float), optional
    :param Wf: Force weighting matrix for the interface impact points
    :type Wf: 2D matrix (float), optional
    :param sort_matrix: Sort transformation matrices
    :type sort_matrix: bool, optional

    Transformed admittance matrix is sorted by increasing grouping number. VP DoFs are ordered in the same manner as 
    provided in the dataframe.
    """

    def __init__(self, ch, refch, vp_ch, vp_refch, Wu = None, Wf = None, sort_matrix = True):
        self.sort_matrix = sort_matrix

        # Load the physical input-output DoFs
        self.Channels = ch
        self.RefChannels = refch

        # Load virtual input-output DoFs and order by grouping number
        self.Virtual_Channels = vp_ch.sort_values(["Grouping"], kind='stable')
        self.Virtual_RefChannels = vp_refch.sort_values(["Grouping"], kind='stable')

        # Load Weighting matrices: if None, no weighting is applied in the transformation
        self.Wu_p = Wu
        self.Wf_p = Wf

        # Define the IDM_U and IDM_F matrix
        self.define_IDM_U()
        self.define_IDM_F()


    def define_IDM_U(self):
        """
        Calculates Ru, Tu and Fu matrices based on the supplied position and orientation of Channels and Virtual
        Channels.
        """
        ov_u, _vps, mask_u = self.find_overlap(self.Channels, self.Virtual_Channels)
        R_all = []
        _Warray = []

        # iterates through all unique virtual points (through grouping)
        for i in range(len(ov_u)):
            # gets the unique VP position
            _posVP = self.Virtual_Channels.iloc[_vps[1][i]][["Position_1","Position_2","Position_3"]].to_numpy()
            # gets defined DoF for specific VP
            _desc = self.Virtual_Channels.loc[self.Virtual_Channels["Grouping"]==_vps[0][i]]["Description"].to_list()
            # gets the current positions
            ov_c = ov_u[i]

            # iterates through all channels corresponding to unique virtual point
            r = np.zeros((len(ov_c), len(_desc)))
            for j, ch in enumerate(ov_c):
                # gets position of the single channel
                _pos = self.Channels.iloc[ch][["Position_1","Position_2","Position_3"]].to_numpy()
                # gets orientation of the single channel
                _dir = self.Channels.iloc[ch][["Direction_1","Direction_2","Direction_3"]].to_numpy()
                # gets quantity type of the single channel (either translational or angular acceleration)
                _type = self.Channels.iloc[ch]['Quantity']

                r[j, :] = _dir @ self.R_matrix_U(_pos - _posVP, _desc, type=_type)
                _Warray.append(self.W_rotational(_pos, _dir, type=_type))

            R_all.append(r)

        # add channels not belonging to any VP
        Ru = block_diag(*R_all, np.eye(np.count_nonzero(mask_u)))

        # sorting of the Ru matrix
        if self.sort_matrix == True:
            # sort on channels
            _ov_u = np.concatenate((np.concatenate(ov_u), np.where(mask_u == 1)[0]))
            Ru = Ru[np.argsort(_ov_u, kind='stable'),:]
            # sort on VPs
            ind_vp = self.Virtual_Channels['Grouping'].to_numpy()
            _ind_vp = np.concatenate((ind_vp, self.Channels.iloc[np.where(mask_u == 1)[0]]['Grouping']))
            Ru = Ru[:,np.argsort(_ind_vp, kind='stable')]

        # definition of weighting matrix
        Wu = np.eye(np.max(Ru.shape))
        if self.Wu_p is not None:
            interfaceDOFs_u = np.where(mask_u == 0)[0]
            Wu[np.ix_(interfaceDOFs_u,interfaceDOFs_u)] = self.Wu_p

        # calculate the Tu, Fu matrices
        Tu = np.linalg.pinv(Ru.T @ Wu @ Ru) @ Ru.T @ Wu
        Fu = Ru @ Tu

        self.Ru = Ru
        self.Wu = Wu
        self.Tu = Tu
        self.Fu = Fu

    def define_IDM_F(self):
        """
        Calculates the Rf, Tf, Ff matrices based on the supplied position and orientation of Reference Channels and
        Reference Virtual Channels.
        """

        ov_f, _vps, mask_f = self.find_overlap(self.RefChannels, self.Virtual_RefChannels)
        
        R_all = []

        # iterates through all unique virtual points (through grouping)
        for i in range(len(ov_f)):
            # gets the unique VP position
            _posVP = self.Virtual_RefChannels.iloc[_vps[1][i]][["Position_1","Position_2","Position_3"]].to_numpy()
            # gets defined DoF for specific VP
            _desc = self.Virtual_RefChannels.loc[self.Virtual_RefChannels["Grouping"]==_vps[0][i]]["Description"].to_list()
            # gets the current positions
            ov_c = ov_f[i]

            # iterates through all impacts corresponding to unique virtual point
            r = np.zeros((len(ov_c), len(_desc)))
            for j, im in enumerate(ov_c):
                # gets position of the single impact
                _pos = self.RefChannels.iloc[im][["Position_1", "Position_2", "Position_3"]].to_numpy()
                # gets orientation of the single impact
                _dir = self.RefChannels.iloc[im][["Direction_1", "Direction_2", "Direction_3"]].to_numpy()

                r[j, :] = (self.R_matrix_F(_pos - _posVP, _desc) @ (_dir).T)

            R_all.append(r)

        # add impacts not belonging to any VP
        Rf = block_diag(*R_all, np.eye(np.count_nonzero(mask_f)))

        # sorting of the Rf matrix
        if self.sort_matrix == True:
            # sort on impacts
            _ov_f = np.concatenate((np.concatenate(ov_f), np.where(mask_f == 1)[0]))
            Rf = Rf[np.argsort(_ov_f),:]
            # sort on VPs
            ind_vpref = self.Virtual_RefChannels['Grouping'].to_numpy()
            _ind_vpref = np.concatenate((ind_vpref, self.RefChannels.iloc[np.where(mask_f == 1)[0]]['Grouping']))
            Rf = Rf[:,np.argsort(_ind_vpref, kind='stable')]

        # definition of weighting matrix            
        Wf = np.eye(np.max(Rf.shape))
        if self.Wf_p is not None:
            interfaceDOFs_f = np.where(mask_f == 0)[0]
            Wf[np.ix_(interfaceDOFs_f,interfaceDOFs_f)] = self.Wf_p

        # calculate the Tf, Ff matrices
        Tf = np.linalg.pinv(Wf) @ Rf @ np.linalg.pinv(Rf.T @ np.linalg.pinv(Wf) @ Rf)
        Ff = Rf @ Tf.T

        self.Rf = Rf
        self.Wf = Wf
        self.Tf = Tf
        self.Ff = Ff

    @staticmethod
    def R_matrix_U(pos, desc, type="Acceleration"):
        """
        Calculate Ru matrix based on the channel position/orientation and sensor type.

        :param pos: Position of the channel.
        :type pos: array(float)
        :param type: Type of the channel (i.e. Acceleration or Angular Acceleration).
        :type pos: string, optional
        :returns: Ru matrix
        """

        rx, ry, rz = pos

        if type == "Angular Acceleration":
            _R = np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        else:
            _R = np.asarray([[1, 0, 0, 0, rz, -ry, rx, 0, 0, 0, ry * rz, -rz * ry, rx * ry, rx * rz, 0, 0, 0, 0],
                             [0, 1, 0, -rz, 0, rx, 0, ry, 0, -rx * rz, 0, rz * rx, 0, 0, ry * rz, ry * rx, 0, 0],
                             [0, 0, 1, ry, -rx, 0, 0, 0, rz, rx * ry, -ry * rx, 0, 0, 0, 0, 0, rz * rx, rz * ry]])

        # isolating desired DoF
        columns_ = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz', 'ex', 'ey', 'ez', 'tx', 'ty', 'tz', 'sxy', 'sxz', 'syz', 'syx',
                    'szx', 'szy']
        _R = np.asarray(pd.DataFrame(_R, columns=columns_)[desc])

        return _R

    @staticmethod
    def W_rotational(pos, dir, type="Angular Acceleration"):
        """
        Defines the weighting matrix based on the location of rotational accelerometer. 
        If translational accelerometer is used, identity matrix of appropriate size is 
        defined.

        :param pos: Position of the channel.
        :type pos: array(float)
        :param dir: Direction of the channel
        :type dir: array(float)
        :param type: Type of the channel (i.e. Acceleration or Angular Acceleration)
        :type type: str, optional
        """

        rx, ry, rz = pos

        if type == "Angular Acceleration":
            c = np.where(np.asarray(dir) != 0)[1][0]
            if c == 0:
                _W = np.sqrt(rz ** 2 + ry ** 2) ** 2
            elif c == 1:
                _W = np.sqrt(rz ** 2 + rx ** 2) ** 2
            elif c == 2:
                _W = np.sqrt(ry ** 2 + rx ** 2) ** 2
                
        else:
            _W = 1

        return _W

    @staticmethod
    def R_matrix_F(pos, desc):
        """
        Calculates Rf matrix based on the reference channel position/orientation.

        :param pos: Position of the reference channel relative to the virtual point.
        :type pos: array(float)
        :returns: Rf matrix
        """

        rx, ry, rz = pos

        _R = np.asarray([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, -rz, ry],
                         [rz, 0, -rx],
                         [-ry, rx, 0],
                         [rx, 0, 0],
                         [0, ry, 0],
                         [0, 0, rz],
                         [0, -rx * rz, rx * ry],
                         [ry * rz, 0, -ry * rx],
                         [-rz * ry, rz * rx, 0],
                         [rx * ry, 0, 0],
                         [rx * rz, 0, 0],
                         [0, ry * rz, 0],
                         [0, ry * rx, 0],
                         [0, 0, rz * rx],
                         [0, 0, rz * ry]])

        # isolating desired DoF
        columns_ = ['fx','fy','fz','mx','my','mz','ex','ey','ez','tx','ty','tz','sxy','sxz','syz','syx','szx','szy']
        _R = np.asarray(pd.DataFrame(_R.T, columns=columns_)[desc]).T
        
        return _R

    @staticmethod
    def find_overlap(channelsA, channelsB):
        """
        Finds an overlap of grouping number between two DataFrames.

        :param channelsA: First dataset
        :type channelsA: pd.DataFrame
        :param channelsB: Second dataset
        :type channelsB: pd.DataFrame
        :return: overlap, unique_index, overlap_mask
        """

        # Get the grouping numbers from DataFrames
        _group_A = channelsA["Grouping"].to_numpy()
        _group_B = channelsB["Grouping"].to_numpy()

        # Find overlap between the two datasets
        _overlap = []
        for a in np.unique(_group_B):
            _overlap.append(np.where(_group_A == a)[0])

        # Sort channels not included in the transformation
        mask = np.in1d(_group_A, np.unique(_group_B), invert=True).astype(int)

        # Sort VP by when they appear in the dataframe
        unique_VP = np.unique(_group_B, return_index=True)

        return _overlap, unique_VP, mask

    @staticmethod
    def find_group(gr, gr_list):
        """
        Get a grouping overlap between two DataFrames.

        :param gr: Grouping number
        :type gr: List of integers
        :param gr_list: A list of grouping numbers
        :type gr_list: list
        :return: overlap_mask
        """

        _overlap = []
        for a in np.unique(gr):
            _arr_file = np.array(np.where(gr_list == a)).reshape(-1)
            _overlap.append(_arr_file)
        return np.concatenate(_overlap, axis=0)


    def apply_VPT(self, freq, FRF):
        """
        Applies the Virtual Point Transformation on the FRF matrix.

        :param freq: Frequency vector
        :type freq: array(float)
        :param FRF: A matrix of Frequency Response Functions FRFs [f,out,in].
        :type FRF: array(float)
        """
        _Y_vpt = self.Tu @ FRF @ self.Tf

        self.vptData = _Y_vpt
        self.freq = freq
        self.FRF = FRF

    def consistency(self, grouping, ref_grouping):
        """
        Evaluates VP consistency indicators based on the supplied grouping numbers.

        :param grouping: Grouping number of the VP.
        :type grouping: float
        :param ref_grouping: Grouping number of the reference VP.
        :type ref_grouping: float
        """

        # get all groupings from the vpt
        _ch_all = self.Channels.Grouping.to_numpy()
        _chVP_all = self.Virtual_Channels.Grouping.to_numpy()

        _Rch_all = self.RefChannels.Grouping.to_numpy()
        _RchVP_all = self.Virtual_RefChannels.Grouping.to_numpy()

        # extract the grouping mask
        ind_ch = self.find_group(grouping, _ch_all)
        ind_Rch = self.find_group(ref_grouping, _Rch_all)

        ind_NotRemovedChannels = np.nonzero(np.diag(self.Wu))[0]
        ind_NotRemovedImpacts = np.nonzero(np.diag(self.Wf))[0]
        
        ind_NotRemovedChannels_Grouping = sorted(np.intersect1d(ind_NotRemovedChannels,ind_ch))
        ind_NotRemovedImpacts_Grouping = sorted(np.intersect1d(ind_NotRemovedImpacts,ind_Rch))
        
        # Calculate sensor consistency
        sub_Y = np.transpose(self.FRF,(1,2,0))[ind_NotRemovedChannels_Grouping, :, :][:, ind_NotRemovedImpacts_Grouping, :]
        sub_Fu = self.Fu[ind_NotRemovedChannels_Grouping, :][:, ind_NotRemovedChannels_Grouping]

        u_f = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)
        u = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # print(i)
            # filtered response
            u_f[:, :, i] = sub_Fu @ sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))
            # initial response
            u[:, :, i] = sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))

        self.u_f = u_f[:,0,:]
        self.u = u[:,0,:]

        # Calculate overall sensor consistency indicator
        self.overall_sensor = norm(self.u_f,axis = 0) / norm(self.u,axis = 0)


        # Calculate specific sensor consistency indicator
        specific_sensor = []
        for i in range(self.u.shape[0]):
            specific_sensor.append(coh_frf(self.u_f[i, :], self.u[i, :]))

        self.specific_sensor = np.asarray(specific_sensor)


        # Calculate impact consistency
        sub_Y = np.transpose(self.FRF,(1,2,0))[ind_NotRemovedChannels_Grouping, :, :][:, ind_NotRemovedImpacts_Grouping, :]
        sub_Ff = self.Ff[ind_NotRemovedImpacts_Grouping, :][:, ind_NotRemovedImpacts_Grouping]

        y_f = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)
        y = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            y_f[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i] @ sub_Ff).T
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

    """
    Frequency-dependend weighting matrix - to be implemented in the pyFBS with next release

    Wu = block_diag(*_Warray)
    Wu = block_diag(Wu, np.eye(len(np.where(mask_u != 0)[0])))
    self.Wu = Wu

    # Wu_f is a 4D numpy array for each input set you use where -j refers to the freq input W[i,:,:,j]
    # n_imp... number of impacts
    # n_out... number of outputs
    # n_freq.. number of freqs

    n_out = self.Y.Data.shape[0]
    n_imp = self.Y.Data.shape[1]
    n_freq = self.Y.Data.shape[2]

    Wu_f = np.zeros((n_imp, n_out, n_freq))
    Tu_f = np.zeros((n_imp, self.Ru.shape[1], n_out, n_freq))
    # Fu_f = np.zeros((n_imp, n_out , n_out , 1000))

    for _f in tqdm(range(n_freq)):
        for _i in range(n_imp):
            _tW = block_diag(*np.abs(self.Y.Coherence[:, _i, _f])) ** 2
            # print(_tW.shape)
            # for _d,diag_val in enumerate(np.diag(_tW)):
            #    if _d in [9,10,11,21,22,23,33,34,35]:
            #        _tW[_d,_d] = 0#sigmoid(self.Y.Freqs[_f])

            W_sum = Wu + _tW
            Wu_f[_i, :, _f] = np.diag(W_sum)

            Tu = np.linalg.pinv(Ru.T @ W_sum @ Ru) @ Ru.T @ W_sum
            # Fu = Ru @ Tu
            Tu_f[_i, :, :, _f] = Tu
            # Fu_f[_i, :, :, _f] = Fu

    self.Tu = Tu
    self.Wu_f = Wu_f

    self.Tu_f = Tu_f
    self.Fu_f = Fu_f
    """