import pyuff
import math
from .utility import *
import os
import requests
import shutil
import h5py    
from tqdm import tqdm

LAB_FOLDER = "lab_testbench"
LAB_FILES = {"FEM": ["A.full", "A.rst", "AB.full", "AB.rst", "AB_overlay.full", "AB_overlay.rst", "AB_parent.full", "AB_parent.rst", "B.full", "B.rst"],
                 "STL": ["A.stl", "B.stl", "AB.stl"],
                 "Measurements": ["AM_Measurements.xlsx","ammeasurements.xlsx", "coupling_example.xlsx", "decoupling_example.xlsx",
                                  "decoupling_example_SVT.xlsx", "TPA_synt.xlsx", "Y_A.p", "Y_B.p", "Y_AB.p", "expansion_synt_example.xlsx"]}

AUTOMOTIVE_FOLDER = "automotive_testbench"
AUTOMOTIVE_FILES = {"FEM": ["EM.full", "EM.rst", "RM.full", "RM.rst", "TM.full", "TM.rst"],
                 "STL": ["engine_mount.stl", "receiver.stl", "roll_mount.stl", "shaker_only.stl", "source.stl",
                         "transmission_mount.stl", "ts.stl"],
                 "Measurements": ["A.p", "A.xlsx", "AB_ref.p", "AB_ref.xlsx", "BTS.p", "BTS.xlsx", "B_ref.p",
                                  "ODS.p", "ODS.xlsx", "TS.p", "TS.xlsx", "TS.xlsx", "frame_rubbermounts.p",
                                  "frame_rubbermounts_sourceplate.p", "modal.xlsx"]}

RM_FOLDER = "rubber_mount"
RM_FILES = {"STL": ["model.stl"],
                 "Measurements": ["data.xlsx", "freq.npy", "Y_A_B.npy", "Y_AJB.npy"]}

def load_uff_file_PAK(uff_file_data,uff_file_output,uff_file_input,chn_input = False):
    """
    Loads an Universal File Format .uff file from PAK system and parses the data in arrays and DataFrames

    :param uff_file_data: A filename of .uff file containing information on FRFs
    :type uff_file_data: str
    :param uff_file_output: A filename of .uff file containing information on channels
    :type uff_file_output: str
    :param uff_file_input: A filename of .uff file containing information on reference channels
    :type uff_file_input: str
    :returns: frequency vector, FRF matrix, channel DataFrame, impact DataFrame, sensor DataFrame
    """

    uff_file_out = pyuff.UFF(uff_file_output)
    data_output = uff_file_out.read_sets()

    uff_file_in = pyuff.UFF(uff_file_input)
    data_input = uff_file_in.read_sets()

    uff_file = pyuff.UFF(uff_file_data)
    data = uff_file.read_sets()

    chn_dof = len(data_output["x"]) 
    
    # assuming tri-axial accelerometers
    if chn_input:
        _value = 1
        chn_dof *= 3
    else:
        _value = 3
    imp_dof = len(data_input["x"])

    Directions = {0: "None", 1: "+X", 2: "+Y", 3: "+Z", -1: "-X", -2: "-Y", -3: "-Z"}
    Directions_array = {0: "None", 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1], -1: [-1, 0, 0], -2: [0, -1, 0],
                        -3: [0, 0, -1]}

    freq = data[0]["x"]
    FRF = np.zeros((len(freq), chn_dof, imp_dof), dtype=complex)

    out_resp = np.zeros((chn_dof, 3))
    in_resp = np.zeros((imp_dof, 3))

    out_N = [""] * chn_dof
    in_N = [""] * imp_dof

    i = 0
    for _out in range(chn_dof):
        for _in in range(imp_dof):
            Node_Number = data[i]['rsp_node']
            Name = 'S' + str(math.ceil(Node_Number / _value)) + " " + Directions[data[i]['rsp_dir']]
            
            out_resp[_out, :] = Directions_array[data[i]['rsp_dir']]
            in_resp[_in, :] = Directions_array[data[i]['ref_dir']]

            Node_Number = data[i]['ref_node']
            RefName = 'H' + str(Node_Number) + " " + Directions[data[i]['ref_dir']]

            FRF[:, _out, _in] = data[i]["data"]

            out_N[_out] = Name
            in_N[_in] = RefName

            i += 1

    # parse channel data
    columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                     "Grouping", "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2",
                     "Direction_3"]

    df = pd.DataFrame(columns=columns_chann)

    for _out in range(int(chn_dof / 3)):
        for i in range(3):
            out_dir = out_resp[_out * 3 + i]
            out_pos = [data_output["x"][_out], data_output["y"][_out], data_output["z"][_out]]
            data_chn = np.asarray([[out_N[_out * 3 + i], None, None, out_N[_out * 3 + i].split(" ")[1], None, None,
                                    None, None, None, out_pos[0],
                                    out_pos[1], out_pos[2], out_dir[0], out_dir[1], out_dir[2]]])

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = pd.concat([df, df_row], ignore_index=True)

    if chn_input:
        df_chn = df
        df_acc = generate_sensors_from_channels(df_chn)
    else:
        df_chn = df
        df_acc = None
        
        


    # parse impact data
    columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                     "Grouping", "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2",
                     "Direction_3"]

    df = pd.DataFrame(columns=columns_chann)

    for _in in range(imp_dof):
        in_pos = [data_input["x"][_in], data_input["y"][_in], data_input["z"][_in]]
        in_dir = in_resp[_in]

        data_chn = np.asarray(
            [[in_N[_in].split(" ")[0], None, None, in_N[_in].split(" ")[1], None, None, None, None, None, in_pos[0],
              in_pos[1], in_pos[2], in_dir[0], in_dir[1], in_dir[2]]])

        df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
        df = pd.concat([df, df_row], ignore_index=True)

    df_imp = df


    return freq,FRF,df_chn,df_imp,df_acc


def download_automotive_testbench(overwrite=False):
    """

    """
    folder_name = AUTOMOTIVE_FOLDER #"automotive_testbench"

    url_automotive = r'https://gitlab.com/pyFBS/pyFBS_data/-/raw/master/automotive_testbench/'

    url_a_sub = AUTOMOTIVE_FILES #{"FEM": ["EM.full", "EM.rst", "RM.full", "RM.rst", "TM.full", "TM.rst"],
                # "STL": ["engine_mount.stl", "receiver.stl", "roll_mount.stl", "shaker_only.stl", "source.stl",
                #         "transmission_mount.stl", "ts.stl"],
                # "Measurements": ["A.p", "A.xlsx", "AB_ref.p", "AB_ref.xlsx", "BTS.p", "BTS.xlsx", "B_ref.p",
                #                  "ODS.p", "ODS.xlsx", "TS.p", "TS.xlsx", "TS.xlsx", "frame_rubbermounts.p",
                #                  "frame_rubbermounts_sourceplate.p", "modal.xlsx"]}

    # remove folder if overwrite
    if os.path.isdir(folder_name) and overwrite:
        shutil.rmtree(folder_name)

    # create folder
    if not (os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    for sub_dir in url_a_sub:
        print("Downloading %s files" % sub_dir)

        # create a subdirectory
        if not (os.path.isdir(folder_name + os.sep + sub_dir)):
            os.mkdir(folder_name + os.sep + sub_dir)

        for filename in tqdm(url_a_sub[sub_dir]):
            # check if it is file
            if not (os.path.isfile(folder_name + os.sep + sub_dir + os.sep + '%s' % filename)):
                # download each file
                url = url_automotive + sub_dir + "/" + filename
                r = requests.get(url)

                # write to local directory
                with open(folder_name + os.sep + sub_dir + os.sep + '%s' % filename, 'wb') as fout:
                    fout.write(r._content)

def download_lab_testbench(overwrite=False):
    """
    Download laboratory testbench files

    """
    folder_name = LAB_FOLDER #"lab_testbench"

    url_lab = r'https://gitlab.com/pyFBS/pyFBS_data/-/raw/master/lab_testbench/'

    url_l_sub = LAB_FILES #{"FEM": ["A.full", "A.rst", "AB.full", "AB.rst", "B.full", "B.rst"],
                # "STL": ["A.stl", "B.stl", "AB.stl"],
                # "Measurements": ["AM_Measurements.xlsx","ammeasurements.xlsx", "coupling_example.xlsx", "decoupling_example.xlsx",
                #                  "TPA_synt.xlsx", "Y_A.p", "Y_B.p", "Y_AB.p"]}

    # remove folder if overwrite
    if os.path.isdir(folder_name) and overwrite:
        shutil.rmtree(folder_name)

    # create folder
    if not (os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    for sub_dir in url_l_sub:
        print("Downloading %s files" % sub_dir)

        # create a subdirectory
        if not (os.path.isdir(folder_name + os.sep + sub_dir)):
            os.mkdir(folder_name + os.sep + sub_dir)

        for filename in tqdm(url_l_sub[sub_dir]):
            # check if it is file
            if not (os.path.isfile(folder_name + os.sep + sub_dir + os.sep + '%s' % filename)):
                # download each file
                url = url_lab + sub_dir + "/" + filename
                r = requests.get(url)
                #print(url)

                # write to local directory
                with open(folder_name + os.sep + sub_dir + os.sep + '%s' % filename, 'wb') as fout:
                    fout.write(r._content)

def download_rubber_mount(overwrite=False):
    """
    Download rubber mount files

    """
    folder_name = RM_FOLDER #"rubber_mount"

    url_lab = r'https://gitlab.com/pyFBS/pyFBS_data/-/raw/master/rubber_mount/'

    url_l_sub = RM_FILES 

    # remove folder if overwrite
    if os.path.isdir(folder_name) and overwrite:
        shutil.rmtree(folder_name)

    # create folder
    if not (os.path.isdir(folder_name)):
        os.mkdir(folder_name)

    for sub_dir in url_l_sub:
        print("Downloading %s files" % sub_dir)

        # create a subdirectory
        if not (os.path.isdir(folder_name + os.sep + sub_dir)):
            os.mkdir(folder_name + os.sep + sub_dir)

        for filename in tqdm(url_l_sub[sub_dir]):
            # check if it is file
            if not (os.path.isfile(folder_name + os.sep + sub_dir + os.sep + '%s' % filename)):
                # download each file
                url = url_lab + sub_dir + "/" + filename
                r = requests.get(url)
                #print(url)

                # write to local directory
                with open(folder_name + os.sep + sub_dir + os.sep + '%s' % filename, 'wb') as fout:
                    fout.write(r._content)

def load_hdf_FRFs(file_name):
    """
    Loads HDF file format - Math (FRFs from Modal test), when exported from Dewesoft 2021.4 software.

    :param uff_file_data: A filename of the .hdf file containing information of FRFs
    :type uff_file_data: str
    """

    f = h5py.File(file_name,'r+')

    keys = []
    for key in f.keys():
        keys.append(key)

    arr_out = []
    arr_in = []

    for key in keys[::2]:
        _file = key.split("_")
        name_real = key
        name_imag = key + "_2"
        
        ch_out = _file[2][:-2]
        ch_in = _file[3][:-2]
        
        arr_out.append(int(ch_out))
        arr_in.append(int(ch_in))
        
    _out = np.max(arr_out) - np.min(arr_out) + 1
    _in = np.max(arr_in) - np.min(arr_in) + 1
    _f = len(f[keys[0]][()][0])

    Y = np.zeros((_f,_out,_in),dtype = complex)
    for i,key in enumerate(keys[::2]):
        
        _file = key.split("_")
        name_real = keys[2*i+0]
        name_imag = keys[2*i+1] 
        
        print(name_real)
        print(name_imag)
        
        
        ch_out = _file[2][:-2]
        ch_in = _file[3][:-2]
        
        resp = (f[name_real][()]+f[name_imag][()]*1j).T
        Y[:,int(ch_out)-1,int(ch_in)-1] = resp[:,0]
    
    return Y, keys 
    
def load_hdf_AI(file_name,group_name = "AI"):
    """
    Loads HDF file format - Analog Input, when exported from Dewesoft software.

    :param uff_file_data: A filename of the .hdf file containing information of FRFs
    :type uff_file_data: str
    """
    
    f = h5py.File(file_name,'r+')

    group = f[group_name]

    keys = []
    for key in group.keys():
        keys.append(key)
    
    acc = []
    for i,chn in enumerate(group.keys()): 
        group[chn][()]
        acc.append(group[chn][()])
        
    acc = np.asarray(acc)
    
    return acc, keys