import os
import mne
import pickle as pkl
import numpy as np

def create_forward_model(sampling='ico3', info=None, verbose=True, fixed_ori=True):
#def create_forward_model(sampling='ico4', info=get_info(kind='easycap-M1'), verbose=0, fixed_ori=True):
    ''' Create a forward model using the fsaverage template form freesurfer.
    函数的主要目的是创建一个前向模型,该模型基于Freesurfer的fsaverage模板,前向模型通常用于将脑源活动映射到传感器空间。
    主要功能是下载fsaverage主题，为该主题创建源空间和前向模型，然后对前向模型进行必要的调整，以适应特定的EEG或MEG数据。这是在脑源分析中的重要步骤，以将脑源活动映射到传感器空间。
    Parameters:
    ----------
    sampling : str, 字符串,前向模型的下采样级别。通常建议使用'ico3'（小）、'ico4'（中）或'ico5'（大）作为采样级别的选择
        the downsampling of the forward model. 
        Recommended are 'ico3' (small), 'ico4' (medium) or 
        'ico5' (large).
        一些推荐的分割级别（subdivisions）和相应的源空间设置，以提供不同级别的解剖细节和计算效率。这些级别是基于对正二十面体（Icosahedron）和八面体（Octahedron）进行划分而得出的。以下是其中的一些设置及其相关信息：'oct5',每半球的源数量：1026;源间距离：9.9毫米;每个源的表面积：97平方毫米.
'ico4':每半球的源数量：2562;源间距离：6.2毫米;每个源的表面积：39平方毫米.
'oct6':每半球的源数量：4098;源间距离：4.9毫米;每个源的表面积：24平方毫米
'ico5':每半球的源数量：10242;源间距离：3.1毫米;每个源的表面积：9.8平方毫米
    info : mne.Info
        info instance which contains the desired 
        electrode names and positions. 
        This can be obtained e.g. from your processed mne.Raw.info, 
        mne.Epochs.info or mne.Evoked.info
        If info is None the Info instance is created where 
        electrodes are chosen automatically from the easycap-M10 layout.0tee  
        info包含所需电极名称和位置信息的info实例。用户可以提供自己的mne.Info对象，也可以为其传递已处理的mne.Raw.info、mne.Epochs.info或mne.Evoked.info对象。如果不提供info对象，函数将创建一个新的info对象，自动选择电极的位置和名称，通常基于easycap-M10布局(easycap-M10 Brainproducts EasyCap with electrodes named according to the 10-05 system)
    fixed_ori : bool
        Whether orientation of dipoles shall be fixed (set to True) 
        or free (set to False).一个布尔值，用于指定磁源方向是否应该被固定（True）还是保持自由（False）。

    Return
    ------
    fwd : mne.Forward,函数的返回值是一个前向模型对象，通常用于将脑源活动投影到传感器空间。这是函数的主要输出。
        The forward model object
    '''

    # Fetch the template files for our forward model获取我们正向模型的模板文件
    fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)#从MNE-Python的数据集中下载了FreeSurfer的fsaverage主题（template）。这个主题用作前向模型的模板。
    subjects_dir = os.path.dirname(fs_dir)#这一行代码获取下载的fsaverage主题所在的目录。

    # The files live in:
    subject = 'fsaverage'#我们将使用fsaverage主题
    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')#定义变量trans，表示变换文件的路径，这个文件用于将源空间（模板脑）和目标空间（实际脑）对齐
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')#定义变量`src`，表示源空间文件的路径。这个文件定义了源的分布
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')#定义变量bem，表示头模型文件的路径，描述了头部的生物电物理特性。

 

    # Create our own info object, see e.g.:
    if info is None:#这一行检查是否提供了`info`参数
        info = get_info()#这一行调用了一个未在代码中定义的函数`get_info()`，该函数用于创建并返回有关电极（或传感器）的信息
        
    # Create and save Source Model
    
    #这一行创建源空间（source space），包括源的位置和排列方式。参数`spacing`定义了源的分布密度，`surface`表示源所在的脑表面，`subjects_dir`指定主题文件的目录。该函数返回源空间的配置
    src = mne.setup_source_space(subject, spacing=sampling, surface='white',
                                        subjects_dir=subjects_dir, add_dist=False,
                                        n_jobs=-1, verbose=True)

    # Forward Model，这一行创建前向模型。它将信息（`info`）、变换文件（`trans`）、源空间（`src`）、头模型（`bem`）等参数传递给函数`mne.make_forward_solution()`，以创建前向模型。参数`eeg=True`表示这是用于EEG数据的前向模型。`mindist`是电极和头部表面之间的最小距离，`n_jobs`表示可用的CPU核心数，`verbose`表示冗长程度。
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                   bem=bem, eeg=True, mindist=5.0, n_jobs=-1,
                                   verbose=True)
    
    if fixed_ori:#这一行检查`fixed_ori`参数，如果为真，执行以下操作。
        # Fixed Orientations，这一行将前向模型中的源方向设置为固定的。它使用`mne.convert_forward_solution()`函数，并传递了一些参数来修改前向模型。
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                            use_cps=True, verbose=True)

    return fwd,src#返回创建的前向模型对象（`fwd`）。


def get_info(kind='standard_1020', sfreq=256):#定义了一个函数`get_info`，它接受两个参数：`kind`（默认为'easycap-M10'）和`sfreq`（默认为1000）
#def get_info(kind='standard_1020', sfreq=250):#定义了一个函数`get_info`，它接受两个参数：`kind`（默认为'easycap-M10'）和`sfreq`（默认为1000）
    ''' Create some generic mne.Info object.
    创建一个MNE-Python中的通用`mne.Info`对象的函数。这个对象包含了有关数据采集系统（例如脑电图或脑磁图系统）的信息。
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

    montage = mne.channels.make_standard_montage(kind)#这一行根据提供的`kind`参数，使用`mne.channels.make_standard_montage()`函数创建了一个标准的电极排列（电极帽布局），`montage`是一个MNE-Python的Montage对象，它包含有关电极的名称和位置信息。
    # 指定 standard_1020 电极帽中的60个电极标签
    selected_labels = ['Fp1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'Fp2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'PO8', 'PO4', 'O2']
    # 使用指定的电极标签创建 mne.Info 对象
    info = mne.create_info(selected_labels, sfreq, ch_types=['eeg']*len(selected_labels), verbose=True)
    #info = mne.create_info(montage.ch_names, sfreq, ch_types=['eeg']*len(montage.ch_names), verbose=True)
    #info = mne.create_info(montage.ch_names, sfreq, ch_types=['eeg']*len(montage.ch_names), verbose=0)
    #使用`mne.create_info()`函数创建了一个`mne.Info`对象。这个对象包括了有关数据采集系统的信息，包括通道名称、采样频率和通道类型。具体来说： `montage.ch_names`包含了电极的名称，这些名称来自于上一步创建的Montage对象。 `sfreq`表示采样频率。
    #定义了通道类型，这里将通道类型设置为'eeg'，并根据Montage对象中电极的数量重复了相应的次数。
    #info.set_montage(kind)#这一行将前面创建的Montage对象应用到Info对象中，以确保通道位置的一致性。
    # 将 Montage 应用到 Info 对象中
    info.set_montage(montage)
    return info#返回创建的`mne.Info`对象，其中包含了有关数据采集系统的信息