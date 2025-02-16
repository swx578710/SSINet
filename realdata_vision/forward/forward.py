import os
import mne
import pickle as pkl
import numpy as np

def create_forward_model(sampling='ico3', info=None, verbose=0, fixed_ori=True):
    ''' Create a forward model using the fsaverage template form freesurfer.
    
    Parameters:
    ----------
    sampling : str
        the downsampling of the forward model. 
        Recommended are 'ico3' (small), 'ico4' (medium) or 
        'ico5' (large).
    info : mne.Info
        info instance which contains the desired 
        electrode names and positions. 
        This can be obtained e.g. from your processed mne.Raw.info, 
        mne.Epochs.info or mne.Evoked.info
        If info is None the Info instance is created where 
        electrodes are chosen automatically from the easycap-M10 
        layout.
    fixed_ori : bool
        Whether orientation of dipoles shall be fixed (set to True) 
        or free (set to False).

    Return
    ------
    fwd : mne.Forward
        The forward model object
    '''

    # 获取正向模型的模板文件。 fsaverage是一个基于40个真实大脑MRI扫描组合的模板大脑
    fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

 

    # Create our own info object, see e.g.:
    if info is None:
        info = get_info()
        
    # Create and save Source Model
    src = mne.setup_source_space(subject, spacing=sampling, surface='white',
                                        subjects_dir=subjects_dir, add_dist=False,
                                        n_jobs=-1, verbose=verbose)

    # Forward Model
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1,
                                    verbose=verbose)
    if fixed_ori:
        # 固定偶极子方向
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                            use_cps=True, verbose=verbose)

    return fwd


def get_info(kind='easycap-M1', sfreq=256):
    ''' Create some generic mne.Info object.
    创建一个MNE中的通用'mne.Info'对象的函数。这个函数包含了有关数据采集系统的信息
    Parameters
    ----------
    kind : str
        kind, for examples see:
            https://mne.tools/stable/generated/mne.channels.make_standard_montage.html#mne.channels.make_standard_montage

    Return
    ------
    info : mne.Info
        The mne.Info object
    '''
    # https://mne.tools/stable/generated/mne.create_info.html#mne.create_info
    # https://mne.tools/stable/auto_tutorials/simulation/plot_creating_data_structures.html

    montage = mne.channels.make_standard_montage(kind)
    selected_labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
    info = mne.create_info(selected_labels, sfreq, ch_types=['eeg']*len(selected_labels), verbose=True)
    # info = mne.create_info(montage.ch_name, sfreq, ch_types=['eeg'] * len(montage.ch_name), verbose=0)
    # 使用mne.create_info(函数创建了一个mne. Info对象。这个对象包括了有关数据采集系统的信息，包括通道名称、采样频率和通道类型。
    # 具体来说:montage.ch_names 包含了电极的名称，这些名称来自于上-步创建的Montage对象。sfreq表示采样频率。
    info.set_montage(kind)
    return info
