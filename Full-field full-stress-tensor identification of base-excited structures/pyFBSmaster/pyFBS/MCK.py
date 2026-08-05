from scipy.sparse import linalg,diags
from ansys.mapdl import reader as pymapdl_reader
from numpy.random import randn
from .VPT import VPT

import pandas as pd
import scipy as sp
import numpy as np
from scipy import spatial
from scipy.linalg import block_diag
import pickle
from os import path




class MK_model(object):
    """
    Initialization of the finite element model. Mass and stiffness matrices are imported and also nodes, DoFs and complete mesh of finite elements are defined.
    If parameter ``recalculate`` is ``Ture`` eigenvalues and eigenvectors are calculated. 
    For faster processing by default pickle file is generated where mass and stiffness matrices are stored and also computed eigenvalues, eigenvectors and used number of modes.
    If changes are detected in the mass or stiffness matrix with respect to the stored pickle file, the calculation of eigenvalues and eigenvectors is repeated.
    
    :param rst_file: path of the .rst file exported from Ansys
    :type rst_file: str
    :param full_file: path of the .full file exported from Ansys
    :type full_file: str
    :param no_modes: number of modes to be included in output of the eigenvalue computation
    :type no_modes: int
    :param allow_pickle: if ``True``, pickle file will be generated to store data or will pickle file be used to load data
    :type allow_pickle: bool
    :param recalculate: if ``False`` just mass and stiffness matrices with corresponding nodes and their DoFs will be imported. If ``True`` also the eigenvalue problem will be solved.
    :type recalculate: bool
    :param scale: distance scaling factor
    :type scale: float
    :param read_rst: if ``True`` reads the eigenvalue solution directly from .rst file
    :type read_rst: bool
    """

    def __init__(self, rst_file=None, full_file=None, manual_mass_matrix=None, manual_stifenss_matrix=None, no_modes=100, allow_pickle=True, recalculate=False, scale=1, read_rst=False):
        
        if rst_file and full_file: # check if rest and full files are defined, that mass and stifenss matrices will be importd from there

            rst = pymapdl_reader.read_binary(rst_file)

            # new version of pyansys
            self.nodes = rst.mesh.nodes*scale  # only translational dofs
            self.mesh = rst.grid
            self.mesh.points *= scale
            self.pts = self.mesh.points.copy()

            if no_modes > len(self.nodes):
                self.no_modes = len(self.nodes)
            else:
                self.no_modes = no_modes

            self._all = False

            full = pymapdl_reader.read_binary(full_file)
            self.dof_ref, K_triu, M_triu = full.load_km(sort=True)  # dof_ref: 0-x 1-y 2-z
            self.M = M_triu + sp.sparse.triu(M_triu, 1).T
            self.K = K_triu + sp.sparse.triu(K_triu, 1).T
            self._K = self.K + diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape) # avoid error

            if self.dof_ref[0, 0] != 1:
                self.dof_ref[:, 0] = self.dof_ref[:, 0] - (self.dof_ref[0, 0] - 1)

            if np.max(self.dof_ref[:, 1]) == 5:
                self.rotation_included = True
            elif np.max(self.dof_ref[:, 1]) == 2:
                self.rotation_included = False

            # an option to read directly the .rst file
            if read_rst == False:
                #print("evaluating M and K matrices")
                p_file = '{}.pkl'.format(full_file)
                # check if there is a .pkl file
                same = False
                if allow_pickle and path.exists(p_file):
                
                    same = self.piclke_check(p_file, no_modes)
                    if same:
                        self.M, self.K, self.eig_freq, self.eig_val, self.eig_vec, no_modes = pickle.load(open(p_file, "rb"))
                    # solve the problem
                else:
                    self.eig_freq, self.eig_val, self.eig_vec = self.eig_solve(self.M, self._K, no_modes)

                if same == False or recalculate == True:
                    self.eig_freq, self.eig_val, self.eig_vec = self.eig_solve(self.M, self._K, no_modes)

                    if allow_pickle:
                        pickle.dump([self.M, self.K, self.eig_freq, self.eig_val, self.eig_vec, no_modes],open(p_file, "wb"))
            else: # read from pyansys - from rst file
                # print("Reading RST file")

                self.M += sp.sparse.triu(self.M, 1).T
                self.K += sp.sparse.triu(self.K, 1).T

                self._K = self.K + diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape) # avoid error

                self.eig_freq, self.eig_val, self.eig_vec, self.eig_vec_strain = self.get_values_from_rst(rst)
                if len(self.eig_freq)>=self.no_modes: # truncation of results in .rst file to match the desired number of modes in ``no_modes`` parameter
                    self.eig_freq = self.eig_freq[:self.no_modes]
                    self.eig_val = self.eig_val[:self.no_modes]
                    self.eig_vec = self.eig_vec[:, :self.no_modes]
                    try:
                        self.eig_vec_strain = self.eig_vec_strain[:, :self.no_modes]
                    except: # if the strain is not included in the .rst file, then the ``self.eig_vec_strain`` is just left as an empty array
                        pass
                else:
                    print(f"Parameter ``no_modes`` is set to {self.no_modes}, but the .rst file from Ansys includes {len(self.eig_freq)} natural frequencies and mode shapes. \n \
                        Therefore value of parameter ``no_modes`` is changed to {len(self.eig_freq)}.")
                    self.no_modes = len(self.eig_freq)

        else: # if mass and stifenss matrices are manualy defined
            self.K, self.M  = manual_stifenss_matrix, manual_mass_matrix
            self._K = self.K + diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape) # avoid error
            if no_modes > len(self.K):
                self.no_modes = len(self.K)
            else:
                self.no_modes = no_modes
            self._all = False
            self.rotation_included = False

            p_file = '{}.pkl'.format("mass_stifenss_matrices")
            same = False
            if allow_pickle and path.exists(p_file):
                same = self.piclke_check(p_file, no_modes)
                if same:
                        self.M, self.K, self.eig_freq, self.eig_val, self.eig_vec, no_modes = pickle.load(open(p_file, "rb"))
                    # solve the problem
                if same == False or recalculate == True:
                    self.eig_freq, self.eig_val, self.eig_vec = self.eig_solve(self.M, self._K, no_modes)

                    if allow_pickle:
                        pickle.dump([self.M, self.K, self.eig_freq, self.eig_val, self.eig_vec, no_modes],open(p_file, "wb"))
            else:
                self.eig_freq, self.eig_val, self.eig_vec = self.eig_solve(self.M, self._K, no_modes)

    def piclke_check(self, p_file, no_modes):
        """The function checks if the defined mass and stiffness matrices are the same as were defined in the saved pickle file.

        :param p_file: name of pickle file
        :type p_file: array
        :param no_modes: number of modes to be included in output of the eigenvalue computation
        :type no_modes: int

        :rtype: bool
        """
        _M,_K,_eig_freq,_eig_val,_eig_vec,_no_modes = pickle.load( open(p_file, "rb" ))
        # check if the solution is the same
        if _K.shape == self.K.shape and _M.shape == self.M.shape:
            check_mas  = (_K != self.K).nnz == 0
            check_stif = (_M != self.M).nnz == 0
        else:
            check_mas = False
            check_stif = False
        check_no_modes = _no_modes == no_modes
        same = np.all([check_mas,check_stif,check_no_modes])
        return same
    
    
    def manual_mesh_definition(self, grid, dof_ref):
        """
        Definition of mesh and DoFs for manually inputed mass and stiffness matrices.

        :param grid: grid definition in form of pyvista.PolyData
        :type grid: pyvista
        :param dof_ref: definition of DoFs inside ``MK_model`` in form of 2D matrix, dimensions nx2, where n is the dimension of square mass or stiffness matrix. The first column represents the index of node location, starting with 1, the second column represents direction of this DoF: 0-x, 1-y, 2-z.
        :type dof_ref: array
        """
        self.mesh = grid
        self.nodes = grid.points
        self.pts = grid.points.copy()
        self.dof_ref = dof_ref

    @staticmethod
    def get_values_from_rst(rst):
        """
        Return eigenvalues and eigenvectors for a given rst file.

        :param rst: rst file
        :rtype: (array(float), array(float), array(float))
        """
        eigen_freq = rst.time_values * 2 * np.pi # from Hz to rad/s
        eigen_val = eigen_freq**2
        eigen_vec = []
        eigen_vec_strain = []
        for i in range(len(rst.time_values)):
            nnum, disp = rst.nodal_displacement(i)
            eigen_vec.append(disp.flatten())
            try:
                nnum, strain = rst.nodal_elastic_strain(i)
                eigen_vec_strain.append(strain.flatten())
            except:
                pass
                
        eigen_vec = np.asarray(eigen_vec).T
        try:
            eigen_vec_strain = np.asarray(eigen_vec_strain).T
        except:
            pass

        return (eigen_freq, eigen_val, eigen_vec, eigen_vec_strain)


    @staticmethod
    def eig_solve(mass_mat, stiff_mat, no_modes):
        """
        Find eigenvalues and eigenvectors for given mass matrix ``mass_mat`` and stiffness matrix ``stiff_mat``.

        :param mass_mat: mass matrix
        :type mass_mat: scipy.sparse
        :param stiff_mat: stiffness matrix
        :type stiff_mat: scipy.sparse
        :param no_modes: number of considered modes
        :type no_modes: int
        :return:
        :rtype: (array(float), array(float), array(float))
        """
        try:
            eigen_val, eigen_vec = sp.sparse.linalg.eigsh(stiff_mat, k=no_modes, M=mass_mat, sigma=0)
        except np.linalg.LinAlgError:
            # sometimes eigenvalue problems can not be solved using sparse configuration, especially for small analytical systems
            eigen_val, eigen_vec = sp.linalg.eig(stiff_mat, mass_mat)

        eigen_val.sort()
        eigen_freq = np.sqrt(np.abs(np.real(eigen_val)))  #/(2*np.pi)
        return (eigen_freq, eigen_val, eigen_vec)

    def find_nearest_locations(self, points, **kwargs):
        """
        This function finds the nearest coordinate locations of defined points array in the corresponding MK model mesh.

        :param points: nodal coordinates of points in 3D space
        :type points: array(float)
        :return: Selected nodes by index and by id regarding the dense mesh
        :rtype: (array(int), array(int))

        """
        _index = []
        for _loc_i in points:
            _index.append(self.mesh.find_closest_point(_loc_i, **kwargs))
        return np.array(_index)

    @staticmethod
    def data_preparation(df, n_dim = 3):
        """
        Returns unique locations of all nodal coordinates in ``df`` and all directions for each node.

        :param df: data frame of locations and corresponding directions 
        :type df: pandas.DataFrame
        :param n_dim: number of dimensions in FEM model 
        :type n_dim: int
        :return: unique nodal coordinates and directions for each node
        :rtype: (array(float), array(int))
        """
        nodes = df[["Position_1", "Position_2", "Position_3"]].values.astype(float)
        directions = df[["Direction_1", "Direction_2", "Direction_3"]].values.astype(float)[:,:n_dim]       

        unique_nodes = nodes[np.sort(np.unique(nodes, axis=0, return_index=True)[1])]
        direction_nodes = []
        for node in unique_nodes:
            loc = np.where((nodes == node).all(axis=1))
            direction_nodes.append(directions[loc])
        return unique_nodes, direction_nodes

    def loc_definition(self, node_index):
        """
        DoF index generation for the node index in the global model.

        :param point_index: response/excitation node index in the global model (starting with 1)
        :type response_point: int or array(int)
        :return: DoF indices corresponding to the input point indices 
        :rtype: int
        """
        node_index = np.asarray([node_index]).ravel()
        return np.array([np.argwhere(self.dof_ref[:,0] == _)[:3] for _ in node_index]).ravel()

    def update_locations_df(self,df,scale = 1):
        """
        Update locations in data frame ``df`` to nearest nodal locations of the finite element model.
        Directions remain the same.

        :param df: data frame of locations, for which the nearest locations in the numerical model will be found.
        :type df: pandas.DataFrame
        :return: updated data frame
        :rtype: pandas.DataFrame
        """
        _df = df.copy(deep = True).reset_index()
        _loc = _df[["Position_1", "Position_2", "Position_3"]].to_numpy()*scale
        _index = self.find_nearest_locations(_loc)
        for i, _indedex_i in enumerate(_index):
            _df.loc[i, ["Position_1", "Position_2", "Position_3"]] = self.nodes[_indedex_i]
        return _df

    def get_modeshape(self,select_mode):
        """
        Return desired mode shape. 

        :param select_mode: order of mode shape, starting from 0
        :type select_mode: int
        :return: selected modes shape
        :rtype: array(float)
        """
        _modeshape = np.zeros_like(self.nodes)
        _mode = self.eig_vec[:, select_mode]
        _dof_ref = self.dof_ref
        if self.rotation_included: # to skip rotational modeshape
            _mode = np.asarray([val for m, val in enumerate(_mode) if m % (3 * 2) < 3])
            _dof_ref = np.asarray([val for m, val in enumerate(_dof_ref) if m % (3 * 2) < 3])
        for ref, mode in zip(_dof_ref, _mode):
            _modeshape[ref[0] - 1, ref[1]] = mode

        return _modeshape

    def get_modeshape_strain(self,select_mode, direction = "X"):
        """
        Return desired strain mode shape. 

        :param select_mode: order of mode shape, starting from 0
        :type select_mode: int
        :return: selected modes shape
        :rtype: array(float)
        """
        STRAIN_DIRECTIONS = ["X", "Y", "Z", "XY", "YZ", "XZ", "EQV"]
        mode_index = STRAIN_DIRECTIONS.index(direction.upper())
        
        _modeshape = self.eig_vec_strain[mode_index::len(STRAIN_DIRECTIONS), select_mode]

        return _modeshape


    def transform_modal_parameters(self, df_channel, df_impact = None, limit_modes = None, modal_damping = None, _all = False, return_channel_only = False, n_dim = 3):
        """
        FEM model reduction to the defined input/output locations and directions.

        :param df_channel: locations and directions of responses where FRFs will be generated
        :type df_channel: pandas.DataFrame
        :param df_impact: locations and directions of impacts where FRFs will be generated
        :type df_impact: pandas.DataFrame
        :param limit_modes: number of modes used for FRF synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        """
        # truncation
        if limit_modes == None:
            no_modes = self.no_modes
        else:
            if limit_modes > len(self.nodes):
                no_modes = len(self.nodes)
            else:
                no_modes = limit_modes

        # eigenvalues
        _eig_val2 = self.eig_freq[:no_modes] ** 2
        # damping

        modal_damping = np.asarray(modal_damping).ravel()
        if modal_damping.all() == None:
            damping = np.zeros(no_modes)
        elif len(modal_damping) == 1:
            damping = np.repeat(modal_damping, no_modes)
        elif len(modal_damping) == no_modes:
            damping = modal_damping
        else: 
            raise Exception('Input for "modal damping" not valid.')
        
        # response DoF
        unique_nodes_chn, direction_nodes_chn = self.data_preparation(df_channel, n_dim)
        index_chn = self.find_nearest_locations(unique_nodes_chn)
        response_points = index_chn + 1
        loc1 = self.loc_definition(response_points)

        # response eigenvector reduction/transformation
        if _all:
            m_p_chan_all = self.eig_vec[:, :no_modes]
            m_p_chan_sensors = block_diag(*direction_nodes_chn) @ self.eig_vec[loc1, :no_modes]
            m_p_chan = np.vstack([m_p_chan_sensors,m_p_chan_all])

        else:
            m_p_chan = block_diag(*direction_nodes_chn) @ self.eig_vec[loc1, :no_modes]
        
        if return_channel_only == True:
            return(_eig_val2, damping, m_p_chan)
        else:    
            # excitation DoF
            unique_nodes_imp, direction_nodes_imp = self.data_preparation(df_impact, n_dim)
            index_imp = self.find_nearest_locations(unique_nodes_imp)
            excitation_points = index_imp + 1
            loc2 = self.loc_definition(excitation_points)
            # excitation eigenvector reduction/transformation
            m_p_imp = block_diag(*direction_nodes_imp) @ self.eig_vec[loc2, :no_modes]
            m_p = np.einsum('ij,kj->jik', m_p_chan, m_p_imp)
            return(no_modes, _eig_val2, damping, m_p)

    def FRF_synth_full(self, f_start = 1, f_end = 2000,  f_resolution= 1, frf_type = "receptance"):
        """
        Synthetisation of frequency response functions using the full harmonic method.

        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param frf_type: define calculated FRF type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        """
        
        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start+1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)
        
        freq = np.arange(f_start, f_end, f_resolution)

        omega = 2 * np.pi * _freq

        K_temp = np.array(self._K[np.newaxis]).repeat(len(_freq), axis=0)
        M_temp = np.array(self.M[np.newaxis]).repeat(len(_freq), axis=0)
        FRF_matrix = np.linalg.inv(K_temp - np.einsum("i,ijk->ijk", omega**2, M_temp))

        if frf_type == "receptance":
            _temp = FRF_matrix

        elif frf_type == "mobility":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, (1j*2*np.pi*_freq))

        elif frf_type == "accelerance":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, -(2*np.pi*_freq)**2)

        self.FRF = _temp
        self.freq = freq


    def FRF_synth(self,df_channel,df_impact,f_start = 1, f_end = 2000, f_resolution= 1, limit_modes = None, modal_damping = None, frf_type = "receptance",_all = False, n_dim = 3):
        """
        Synthetisation of frequency response functions using the mode superposition method.

        :param df_channel: locations and directions of responses where FRFs will be generated
        :type df_channel: pandas.DataFrame
        :param df_impact: locations and directions of impacts where FRFs will be generated
        :type df_impact: pandas.DataFrame
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for FRF synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated FRF type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        :param _all: synthetize response at all nodes - can be usefull to animate FRFs
        :type _all, optional: boolean
        :param n_dim: number of DoFs per one node in MK model (default is 3)
        :type n_dim, optional: boolean
        """

        no_modes, _eig_val2, damping, m_p = self.transform_modal_parameters(df_channel = df_channel, df_impact = df_impact, limit_modes = limit_modes, modal_damping = modal_damping, _all = _all, n_dim = n_dim)
        
        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start+1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)
        
        freq = np.arange(f_start, f_end, f_resolution)

        ome = 2 * np.pi * _freq
        ome2 = ome ** 2        
        
        denominator = (_eig_val2[:no_modes, np.newaxis] - ome2) + np.einsum('ij,i->ij',(ome * self.eig_freq[:no_modes, np.newaxis]),(2 * 1j * damping[:no_modes]))

        FRF_matrix = np.einsum('ijk,il->ljk', m_p, 1 / denominator)

        if frf_type == "receptance":
            _temp = FRF_matrix

        elif frf_type == "mobility":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, (1j*2*np.pi*_freq))

        elif frf_type == "accelerance":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, -(2*np.pi*_freq)**2)

        self.FRF = _temp
        self.freq = freq


    def full_DoF_FRF_synth(self, df_imp, df_sen, f_start = 1, f_end = 2000, f_resolution= 1, limit_modes = None, modal_damping = None, frf_type = "receptance", _all = False):
        """
        Generate FRFs on exact location of impacts and sensors by projecting FRFs from three closest nodes in numercial model.
        Modal superpostition method is used for FRF generation.  Gereated are all 3 translations and three rotations for every DoFs.

        :param df_imp: locations and directions of impacts where FRFs will be generated
        :type df_imp: pandas.DataFrame
        :param df_sen: locations and directions of sensors where FRFs will be generated
        :type df_sen: pandas.DataFrame
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for FRF synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated FRF type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        :param _all: synthetize response at all nodes - can be usefull to animate FRFs
        :type _all, optional: boolean
        :param n_dim: number of DoFs per one node in MK model (default is 3)
        :type n_dim, optional: boolean
        """

        imp_coord = np.asarray([df_imp['Position_1'], df_imp['Position_2'], df_imp['Position_3']]).T
        sen_coord = np.asarray([df_sen['Position_1'], df_sen['Position_2'], df_sen['Position_3']]).T
        
        # finding three nearest nodes
        ind_imp= self.find_nearest_locations(imp_coord, n=imp_coord.shape[1])
        ind_sen = self.find_nearest_locations(sen_coord, n=sen_coord.shape[1])
            
        #generating data frame for impacts
        df_imp_ = np.zeros((int(3*3*ind_imp.shape[0]),3)) # assume nine nearest impacts for VPT
        for k in range(self.nodes[ind_imp].shape[0]):
            df_imp_[9*k:9+9*k,:] = np.repeat(np.asarray([[self.nodes[ind_imp][k,:]]][0]),3,axis=1) # assume nine nearest impacts for VPT
        df_imp = pd.DataFrame(data=df_imp_,columns=('Position_1','Position_2','Position_3'))
        df_imp['Direction_1'] = np.tile([1,0,0,1,0,0,1,0,0],self.nodes[ind_imp].shape[0])
        df_imp['Direction_2'] = np.tile([0,1,0,0,1,0,0,1,0],self.nodes[ind_imp].shape[0])
        df_imp['Direction_3'] = np.tile([0,0,1,0,0,1,0,0,1],self.nodes[ind_imp].shape[0])
        df_imp['Grouping'] = np.repeat([np.arange(ind_imp.shape[0])], 9)
        df_imp['Quantity'] = np.tile(np.repeat(['Acceleration'], 9), ind_imp.shape[0])
            
        #generating data frame for channels
        df_chn_ = np.zeros((int(3*3*ind_sen.shape[0]),3)) # assume three nearest sensors (9 channels) for VPT
        for l in range(self.nodes[ind_sen].shape[0]):
            df_chn_[9*l:9+9*l,:] = np.repeat(np.asarray([[self.nodes[ind_sen][l,:]]][0]),3,axis=1) # assume three nearest sensors for VPT
        df_chn = pd.DataFrame(data=df_chn_,columns=('Position_1','Position_2','Position_3'))
        df_chn['Direction_1'] = np.tile([1,0,0,1,0,0,1,0,0],self.nodes[ind_sen].shape[0])
        df_chn['Direction_2'] = np.tile([0,1,0,0,1,0,0,1,0],self.nodes[ind_sen].shape[0])
        df_chn['Direction_3'] = np.tile([0,0,1,0,0,1,0,0,1],self.nodes[ind_sen].shape[0])
        df_chn['Grouping'] = np.repeat([np.arange(ind_sen.shape[0])], 9)
        df_chn['Quantity'] = np.tile(np.repeat(['Acceleration'], 9), ind_sen.shape[0])
        
        # generating FRF
        self.FRF_synth(df_chn,df_imp,f_start, f_end, f_resolution, limit_modes, modal_damping, frf_type,_all)
            
        # generating data frame for impact virtual points
        df_vp_imp_ = np.zeros((int(6*imp_coord.shape[0]),3))
        for ii in range(imp_coord.shape[0]):
            df_vp_imp_[6*ii:6+6*ii,:] = np.asarray([imp_coord[ii,:]]*6)
        df_vp_imp = pd.DataFrame(data=df_vp_imp_,columns=('Position_1','Position_2','Position_3'))
        df_vp_imp['Direction_1'] = np.tile([1,0,0,1,0,0],imp_coord.shape[0])
        df_vp_imp['Direction_2'] = np.tile([0,1,0,0,1,0],imp_coord.shape[0])
        df_vp_imp['Direction_3'] = np.tile([0,0,1,0,0,1],imp_coord.shape[0])
        df_vp_imp['Quantity'] = np.tile(np.repeat(['Acceleration', 'Rotational Acceleration'], 3), imp_coord.shape[0])
        #df_vp_imp['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], imp_coord.shape[0])
        df_vp_imp['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], 6)

        df_vp_imp['Description'] = np.tile(['fx','fy','fz','mx','my','mz'],imp_coord.shape[0])
        
        # generating data frame for channel virtual points
        df_vp_chn_ = np.zeros((int(6*sen_coord.shape[0]),3))
        for jj in range(sen_coord.shape[0]):
            df_vp_chn_[6*jj:6+6*jj,:] = np.asarray([sen_coord[jj,:]]*6)
        df_vp_chn = pd.DataFrame(data=df_vp_chn_,columns=('Position_1','Position_2','Position_3'))
        df_vp_chn['Direction_1'] = np.tile([1,0,0,1,0,0],sen_coord.shape[0])
        df_vp_chn['Direction_2'] = np.tile([0,1,0,0,1,0],sen_coord.shape[0])
        df_vp_chn['Direction_3'] = np.tile([0,0,1,0,0,1],sen_coord.shape[0])
        df_vp_chn['Quantity'] = np.tile(np.repeat(['Acceleration', 'Rotational Acceleration'], 3), sen_coord.shape[0])
        #df_vp_chn['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], sen_coord.shape[0])
        df_vp_chn['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], 6)
        df_vp_chn['Description'] = np.tile(['ux','uy','uz','tx','ty','tz'],sen_coord.shape[0])
        
        # empty array
        FRF_FDoF = np.zeros((self.FRF.shape[0],ind_sen.shape[0]*6,ind_imp.shape[0]*6),dtype=complex)
        
        # apply VPT
        for res_ in df_chn['Grouping'].unique():
            for exc_ in df_imp['Grouping'].unique():
                # Read impacts and VP impacts
                _df_imp = df_imp[df_imp['Grouping'] == exc_]
                _df_vp_imp = df_vp_imp[df_vp_imp['Grouping'] == exc_]
                # Set impacts Group to match responses Group
                #print("a", _df_imp['Grouping'], _df_vp_imp['Grouping'])
                #_df_imp['Grouping'] = res_
                #_df_vp_imp['Grouping'] = res_
                #print("b",_df_imp['Grouping'], _df_vp_imp['Grouping'])

                # Read responses and VP responses
                _df_chn = df_chn[df_chn['Grouping'] == res_]
                _df_vp_chn = df_vp_chn[df_vp_chn['Grouping'] == res_]
                vpt_ = VPT(_df_chn, _df_imp, _df_vp_chn, _df_vp_imp)
                vpt_.apply_VPT(self.freq, self.FRF[:,9*res_:9*res_+9,9*exc_:9*exc_+9])
                FRF_FDoF[:,6*res_:6*res_+6,6*exc_:6*exc_+6] = vpt_.vptData            
        
        return FRF_FDoF


    @staticmethod
    def custom_FRF_synth(eig_freq, eig_vec_chn, eig_vec_imp ,f_start = 1, f_end = 2000, f_resolution= 1, limit_modes = None, modal_damping = None, frf_type = "receptance"):
        """
        Synthetisation of frequency response functions using the mode superposition method.
        FRFs are generated for all combinations of inputed eigen vectors.

        :param eig_freq: eigen frequencies of cinsidered system in unit: rad/s
        :type eig_freq: numpy.array
        :param eig_vec_chn: eigen vectors of channels where FRFs will be generated 
        :type eig_vec_chn: numpy.array
        :param eig_vec_imp: eigen vectors of impacts where FRFs will be generated 
        :type eig_vec_imp: numpy.array
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for FRF synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated FRF type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        """
        if limit_modes == None:
            no_modes = len(eig_freq)
        else:
            no_modes = limit_modes


        modal_damping = np.asarray(modal_damping).ravel()
        if modal_damping.all() == None:
            damping = np.zeros(no_modes)
        elif len(modal_damping) == 1:
            damping = np.repeat(modal_damping, no_modes)
        elif len(modal_damping) == no_modes:
            damping = modal_damping
        else: 
            raise Exception('Input for "modal damping" not valid.')

        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start+1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)
        
        freq = np.arange(f_start, f_end, f_resolution)

        ome = 2 * np.pi * _freq
        ome2 = ome ** 2
        _eig_val2 = eig_freq ** 2

        m_p_chn = eig_vec_chn[:, :no_modes]

        m_p_imp = eig_vec_imp[:, :no_modes]

        m_p = np.einsum('ij,kj->jik', m_p_chn, m_p_imp)
        
        denominator = (_eig_val2[:no_modes, np.newaxis] - ome2) + np.einsum('ij,i->ij',(ome * eig_freq[:no_modes, np.newaxis]),(2 * 1j * damping[:no_modes]))

        FRF_matrix = np.einsum('ijk,il->ljk', m_p, 1 / denominator)

        if frf_type == "receptance":
            _temp = FRF_matrix

        elif frf_type == "mobility":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, (1j*2*np.pi*_freq))

        elif frf_type == "accelerance":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, -(2*np.pi*_freq)**2)

        FRF = _temp

        return freq, FRF


    def add_noise(self,n1 = 2e-2, n2 = 2e-1, n3 = 2e-1 ,n4 = 5e-2):
        """
        Additive noise to synthesized FRFs by random values as per standard normal distribution with defined scaling factors.

        :param n1: amplitude of real part shift scalied with FRF absolute amplitude
        :type n1: float
        :param n2: amplitude of imag part shift scalied with FRF absolute amplitude
        :type n2: float
        :param n3: amplitude of real part shift
        :type n3: float
        :param n4: amplitude of real part shift
        :type n4: float
        """
        rand1 = n1 * np.random.randn(*self.FRF.shape)
        rand2 = n2 * np.random.randn(*self.FRF.shape) * 1j
        rand3 = n3 * np.random.randn(*self.FRF.shape)
        rand4 = n4 * np.random.randn(*self.FRF.shape) * 1j

        noise = np.einsum("ijk,ijk->ijk", np.abs(self.FRF), rand1) + np.einsum("ijk,ijk->ijk", np.abs(self.FRF), rand2) + rand3 + rand4

        self.FRF_noise = self.FRF + noise

