import pickle as pkl#导入 Python 的 pickle 模块，用于序列化和反序列化对象。
import mne#导入 MNE-Python 库，用于处理脑电数据。
from mne.io.constants import FIFF#从 MNE-Python 库导入 FIFF 常量，用于处理文件格式。
import numpy as np#导入 NumPy 库，用于科学计算。
import os#导入 Python 的 os 模块，用于与操作系统交互，例如处理文件路径等。
from joblib import delayed, Parallel#从 joblib 库导入 delayed 和 Parallel，用于并行计算。
from copy import deepcopy#从 Python 的 copy 模块导入 deepcopy 函数，用于深拷贝对象。
import logging#导入 Python 的 logging 模块，用于记录日志。
from time import time#从 Python 的 time 模块导入 time 函数，用于记录时间。
from scipy.stats import pearsonr#从 SciPy 库导入 pearsonr 函数，用于计算皮尔逊相关系数。
from scipy.spatial.distance import cdist#从 SciPy 库导入 cdist 函数，用于计算两组观测之间的距离。
from tqdm.notebook import tqdm#从 tqdm 库导入 tqdm 函数，用于创建进度条。
import matplotlib.pyplot as plt#导入 Matplotlib 库，用于绘图。
from matplotlib.backends.backend_pdf import PdfPages# 从 Matplotlib 库导入 PdfPages，用于将多个图保存到 PDF 文件。
from .. import simulation#导入自定义模块中的 simulation。
from .. import net#导入自定义模块中的 net。

# from .. import Simulation
# from .. import Net
#以下是三个元组，包含了用于判断对象类型的实例
EPOCH_INSTANCES = (mne.epochs.EpochsArray, mne.Epochs, mne.EpochsArray, mne.epochs.EpochsFIF)
EVOKED_INSTANCES = (mne.Evoked, mne.EvokedArray)
RAW_INSTANCES = (mne.io.Raw, mne.io.RawArray)

def load_info(pth_fwd):#load_info 函数用于加载 info 对象，该对象包含了脑电数据的一些信息，如通道名称、频率等。
    with open(pth_fwd + '/info.pkl', 'rb') as file:  
        info = pkl.load(file)
    #print("util loadinfo")
    return info
    
def gaussian(x, mu, sig):#gaussian 函数定义了一个高斯函数，用于生成高斯分布。
    #print("util gaussian")
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def load_leadfield(pth_fwd):#用于加载前向模型（leadfield matrix），如果已经保存为文件，则直接加载，否则通过 load_fwd 函数生成。
    ''' Load the leadfield matrix from the path of the forward model.从前向模型的路径中加载导联场矩阵'''

    if os.path.isfile(pth_fwd + '/leadfield.pkl'):
        with open(pth_fwd + '/leadfield.pkl', 'rb') as file:  
            leadfield = pkl.load(file)
    else:
        fwd = load_fwd(pth_fwd)
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                use_cps=True, verbose=0)
        leadfield = fwd_fixed['sol']['data']
    #print("util load_leadfield")
    return leadfield[0]

def load_fwd(pth_fwd):#load_fwd 函数用于加载前向模型,前向模型描述了从源空间（脑中的潜在活动源）到传感器空间（脑电通道）的映射关系。前向模型通常由 MNE-Python 中的 mne.Forward 类表示。
    fwd = mne.read_forward_solution(pth_fwd + '/fsaverage-fwd.fif', verbose=True)
    #print("util load_fwd")
    return fwd

def get_neighbors(fwd):#函数接受一个前向模型对象 `fwd` 作为参数，其中 `fwd` 是 MNE-Python 中的 `mne.Forward` 对象，包含了源空间的信息。
    """
    其目的是获取源模型中每个偶极子（dipole）的直接邻居。在脑电磁学中，偶极子表示脑中活动的小区域，而源模型则描述了这些偶极子的空间分布。
    Retreive the list of direct neighbors for each dipole in the source model
    Parameters:
    -----------
    fwd : mne.Forward, the mne Forward object 
    Return:
    -------
    neighbors : numpy.ndarray, a matrix containing the neighbor 
        indices for each dipole in the source model

    """
    tris_lr = [fwd['src'][0]['use_tris'], fwd['src'][1]['use_tris']]#分别获取左脑（左半球）和右脑（右半球）的三角形面片信息。
    neighbors = simulations.get_triangle_neighbors(tris_lr)#调用 `simulations` 模块中的 `get_triangle_neighbors` 函数，该函数接受左右脑三角形面片的列表，并返回一个表示直接邻居关系的列表。
    neighbors = np.array([np.array(d) for d in neighbors], dtype='object')#将直接邻居关系的列表转换为 NumPy 数组，其中每个元素是一个包含邻居索引的数组。
    #print("util get_neighbors")
    return neighbors# 最后，函数返回包含直接邻居索引的数组 `neighbors`。

def source_to_sourceEstimate(data, fwd, sfreq=1, subject='fsaverage', 
    simulationInfo=None, tmin=0):
    ''' 
 目的是将源空间的数据转换为 MNE-Python 中的`mne.SourceEstimate`对象.`mne.SourceEstimate`是表示源空间时间序列的对象，通常用于在脑中可视化和分析源活动。
    Takes source data and creates mne.SourceEstimate object
    https://mne.tools/stable/generated/mne.SourceEstimate.html

    Parameters:
    -----------
    data : numpy.ndarray, shape (number of dipoles x number of timepoints), 输入数据，表示源空间的时间序列数据，其形状为 (偶极子数 x 时间点数)。
    pth_fwd : path to the forward model files sfreq : sample frequency, needed
        if data is time-resolved (i.e. if last dim of data > 1)
    sfreq : float, optional是采样频率，如果数据是时域数据（即数据的最后一个维度大于1），则需要提供采样频率。 默认为 1。
    Return:
    -------
    src : mne.SourceEstimate, instance of SourceEstimate.创建的 `mne.SourceEstimate` 对象实例。

    '''
    
    data = np.squeeze(np.array(data))# 对输入的 `data` 进行处理，确保其为 NumPy 数组，并通过 `np.squeeze` 移除数组中的冗余维度。
    if len(data.shape) == 1:#如果数据是一维的，即形状为 (number of dipoles,)，则通过 `np.expand_dims` 在第二个轴上添加一个维度，将其变为二维数组，形状为 (number of dipoles, 1)。
        data = np.expand_dims(data, axis=1)
    
    source_model = fwd['src']# 获取前向模型对象 `fwd` 中的源模型信息。
    number_of_dipoles = int(fwd['src'][0]['nuse']+fwd['src'][1]['nuse'])#计算源模型中使用的偶极子的总数。
    
    if data.shape[0] != number_of_dipoles:#如果数据的第一个维度与偶极子的总数不匹配，进行转置操作，确保数据的维度正确。
        data = np.transpose(data)

    vertices = [source_model[0]['vertno'], source_model[1]['vertno']]#获取左脑和右脑的顶点信息。
    src = mne.SourceEstimate(data, vertices, tmin=tmin, tstep=1/sfreq, 
        subject=subject)# 创建 `mne.SourceEstimate` 对象，其中包含了源活动的时间序列信息。`tmin` 表示起始时间，`tstep` 表示时间步长，`subject` 表示脑模型的主题。
    
    if simulationInfo is not None:#如果提供了仿真信息 `simulationInfo`，将其作为属性添加到 `src` 对象中。
        setattr(src, 'simulationInfo', simulationInfo)

    return src#函数返回创建的 `src` 对象

def eeg_to_Epochs(data, pth_fwd, info=None, parallel=False):
    '''将EEG数据转换为MNE-Python的EpochsArray对象。
    '''
    if info is None:#如果提供了info参数，则使用提供的info，否则从给定的路径pth_fwd加载info。
        info = load_info(pth_fwd)
    
    # if np.all( np.array([d.shape[-1] for d in data]) == data[0].shape[-1]):
    #     epochs = mne.EpochsArray(data, info, verbose=0)
    #     # Rereference to common average if its not the case
    #     if int(epochs.info['custom_ref_applied']) != 0:
    #         epochs.set_eeg_reference('average', projection=True, verbose=0)
    #     epochs = [epochs]
    # else:
    if parallel:
        # 使用并行处理创建EpochsArray对象,函数支持并行处理，如果parallel为True，则使用loky后端并行创建EpochsArray对象。
        epochs = Parallel(n_jobs=-1, backend='loky')(delayed(mne.EpochsArray)(d[np.newaxis, :, :], info, verbose=None) for d in data)
        # if 'eeg' in set(epochs[0].get_channel_types()):
        #     epochs = Parallel(n_jobs=-1, backend='loky')(delayed(epoch.set_eeg_reference)('average', projection=True, verbose=0) for epoch in epochs)
    else:
        #串行创建EpochsArray对象
        epochs = [mne.EpochsArray(d[np.newaxis, :, :], info, verbose=None) for d in data]
    
        # if 'eeg' in set(epochs[0].get_channel_types()):
        #     epochs = [epoch.set_eeg_reference('average', projection=True, verbose=0) for epoch in epochs]
    #print("util eeg_to_Epochs")   
    return epochs# 函数返回包含创建的EpochsArray对象的列表。其中，每个对象代表一个EEG数据样本的Epoch。

def rms(x):
    ''' Calculate the root mean square of some signal x.函数的目的是计算给定信号（数据）x的均方根。
    Parameters
    ----------
    x : numpy.ndarray, list，函数的输入参数。它接受一个NumPy数组或列表，表示输入的信号或数据。
        The signal/data.

    Return
    ------
    rms : float，函数的返回值。它是一个浮点数，代表计算得到的输入信号的均方根值。
    '''
    return np.sqrt(np.mean(np.square(x)))#返回均方根值，计算信号的平方值，然后计算平方值的平均值，然后计算均方根值（rms）。

def unpack_fwd(fwd):
    """ Helper function that extract the most important data structures from the 
    mne.Forward object，定义了一个名为unpack_fwd的辅助函数，用于从mne.Forward对象中提取最重要的数据结构。

    Parameters
    ----------
    fwd : mne.Forward，接受一个mne.Forward对象，即前向模型对象。
        The forward model object

    Return
    ------
    fwd_fixed : mne.Forward，返回一个前向模型对象，其中包含了用于固定偶极方向的信息。
        Forward model for fixed dipole orientations
    leadfield : numpy.ndarray，返回一个NumPy数组，表示导联场（gain matrix）。
        The leadfield (gain matrix)
    pos : numpy.ndarray，返回一个NumPy数组，表示源模型中偶极的位置。
        The positions of dipoles in the source model
    tris : numpy.ndarray，返回一个NumPy数组，表示描述源模型的三角形。
        The triangles that describe the source mmodel
    """
    fwd_fixed = fwd
    if not fwd['surf_ori']:#将fwd_fixed初始化为输入的fwd对象。如果前向模型没有包含固定方向的信息（surf_ori为False），则发出警告
        print("Forward model does not contain fixed orientations - expect unexpected behavior!")
        # fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
        #                                             use_cps=True, verbose=0)

    #从前向模型中提取源的三角形（tris）和导联场（leadfield）
    tris = fwd['src'][0]['use_tris']
    leadfield = fwd_fixed['sol']['data']

    source = fwd['src']#从前向模型中提取源模型信息

    if source[0]["type"] == "vol":#检查源的类型是否为体积源（"vol"）。如果是体积源，从source[0]["rr"]中提取源的三维坐标信息，并将其赋给变量pos
        pos = source[0]["rr"]
    else:# 如果源的类型不是体积源，即为表面源，首先尝试从源信息中获取主题历史ID（subject_his_id）。然后，使用mne.vertex_to_mni函数将左半脑（source[0]）和右半脑（source[1]）的顶点号转换为MNI坐标。如果存在异常，例如找不到主题历史ID，将subject_his_id设置为'fsaverage'，并再次进行转换。     
        try:
            subject_his_id = source[0]['subject_his_id']
            pos_left = mne.vertex_to_mni(source[0]['vertno'], 0, subject_his_id, verbose=0)
            pos_right = mne.vertex_to_mni(source[1]['vertno'],  1, subject_his_id, verbose=0)
        except:
            subject_his_id = 'fsaverage'
            pos_left = mne.vertex_to_mni(source[0]['vertno'], 0, subject_his_id, verbose=0)
            pos_right = mne.vertex_to_mni(source[1]['vertno'],  1, subject_his_id, verbose=0)

        pos = np.concatenate([pos_left, pos_right], axis=0)#将左半脑和右半脑的坐标信息连接成一个数组，并将其赋给变量pos。这个pos数组包含了所有源的三维坐标信息。
    return fwd_fixed, leadfield, pos, tris#返回提取的前向模型对象（fwd_fixed）、导联场（leadfield）、源的位置（pos）和源的三角形（tris）。


def calc_snr_range(mne_obj, baseline_span=(-0.2, 0.0), data_span=(0.0, 0.5)):
    """ Calculate the signal to noise ratio (SNR) range of your mne object.一个用于计算MNE对象中数据的信噪比（SNR）范围的函数
    该函数的目的是通过比较基线时间窗口和数据时间窗口的标准差，计算数据的信噪比范围。这有助于评估在给定时间段内信号相对于噪声的强度。
    Parameters
    ----------
    mne_obj : mne.Epochs, mne.Evoked，`mne.Epochs`或`mne.Evoked`的实例。这是包含MEG/EEG数据的MNE对象。
        The mne object that contains your m/eeg data.
    baseline_span : tuple, list，基线时间窗口的范围。它是一个包含两个元素的元组或列表，指定了基线的开始和结束时间（以秒为单位）。
        The range in seconds that defines the baseline interval.
    data_span : tuple, list，数据（信号）时间窗口的范围。同样，是一个包含两个元素的元组或列表，指定了数据的开始和结束时间（以秒为单位）。
        The range in seconds that defines the data (signal) interval.
    
    Return
    ------
    snr_range : list，返回值，是一个包含SNR值的列表，表示在数据中计算得到的信噪比范围。
        range of SNR values in your data.

    """

    if isinstance(mne_obj, EPOCH_INSTANCES):#判断输入的mne_obj是mne.Epochs的实例还是mne.Evoked的实例。如果是mne.Epochs的实例，则取其平均值作为evoked对象。
        evoked = mne_obj.average()
    elif isinstance(mne_obj, EVOKED_INSTANCES):
        evoked = mne_obj
    else:
        msg = f'mne_obj is of type {type(mne_obj)} but should be mne.Evoked(Array) or mne.Epochs(Array).'
        raise ValueError(msg)
    
    
    # data = np.squeeze(evoked.data)
    # baseline_range = range(*[np.argmin(np.abs(evoked.times-base)) for base in baseline_span])
    # data_range = range(*[np.argmin(np.abs(evoked.times-base)) for base in data_span])
    
    # gfp = np.std(data, axis=0)
    # snr_lo = gfp[data_range].min() / gfp[baseline_range].max() 
    # snr_hi = gfp[data_range].max() / gfp[baseline_range].min()

    # snr = [snr_lo, snr_hi]

    data_base = evoked.copy().crop(*baseline_span)._data
    sd_base = data_base.std(axis=0).mean()#使用crop方法从基线时间窗口中截取数据。data_base是基线数据。然后，计算基线数据的标准差，将其平均。
    data_signal = evoked.copy().crop(*data_span)._data
    sd_signal = data_signal.std(axis=0).max()#使用crop方法从数据时间窗口中截取数据。data_signal是信号数据。然后，计算信号数据的标准差，并取最大值。
    snr = sd_signal/sd_base#计算信噪比（SNR），即信号数据的标准差除以基线数据的标准差。
    #print("util  calc_snr_range") 
    return snr#返回计算得到的SNR值。这个函数通过比较基线数据和信号数据的标准差来估计信噪比，返回一个SNR的数值

def repeat_newcol(x, n):
    ''' Repeat a list/numpy.ndarray x in n columns.将输入的列表或numpy数组 `x` 重复成 `n` 列'''
    out = np.zeros((len(x), n))#创建了一个形状为 `(len(x), n)` 的零矩阵
    for i in range(n):#然后在每列中填充 `x` 的副本。
        out[:,  i] = x
    #print("util repeat_newcol") 
    return np.squeeze(out)#通过 `np.squeeze` 将结果的冗余维度去除。



def get_n_order_indices(order, pick_idx, neighbors):
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.用于执行区域生长算法，给定初始的 `pick_idx`（索引）、邻居关系 `neighbors` 和生长阶数 `order`，该函数通过迭代选择邻居的邻居来实现区域生长。
    '''
    current_indices = np.array([pick_idx])#将初始索引 `pick_idx` 放入 `current_indices` 数组中。

    if order == 0:#如果 `order` 为 0，直接返回 `current_indices`。
        return current_indices

    for _ in range(order):#对于每个生长阶段，将当前索引的邻居连接到 `current_indices`，然后更新 `current_indices`。
        current_indices = np.append(current_indices, np.concatenate(neighbors[current_indices]))
    #print("util get_n_order_indices") 
    return np.unique(np.array(current_indices))#返回生长完成后的唯一索引数组

def gaussian(x, mu, sigma):
    ''' Gaussian distribution function.一个用于计算高斯分布的函数。
    
    Parameters
    ----------
    x : numpy.ndarray, list，表示 x 值的参数，可以是 NumPy 数组或列表。
        The x-value.
    mu : float，表示高斯核均值的参数，应为浮点数。
        The mean of the gaussian kernel.
    sigma : float，表示高斯核标准差的参数，应为浮点数
        The standard deviation of the gaussian kernel.
    Return，给定输入 `x`、均值 `mu` 和标准差 `sigma`，它返回相应高斯分布的概率密度值。
    ------
    '''
    #print("util gaussian") 
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def get_triangle_neighbors(tris_lr):#用于计算三角形网格中每个顶点的直接邻居
    if not np.all(np.unique(tris_lr[0]) == np.arange(len(np.unique(tris_lr[0])))):#检查左半脑三角形网格的索引是否按照预期的方式排列。如果不是，进行以下处理
        for hem in range(2):
            old_indices = np.sort(np.unique(tris_lr[hem]))
            new_indices = np.arange(len(old_indices))
            for old_idx, new_idx in zip(old_indices, new_indices):
                tris_lr[hem][tris_lr[hem] == old_idx] = new_idx

        #print('indices were weird - fixed them.')
    numberOfDipoles = len(np.unique(tris_lr[0])) + len(np.unique(tris_lr[1]))#计算左右两半脑的唯一顶点数量之和
    neighbors = [list() for _ in range(numberOfDipoles)]#创建一个空列表 `neighbors`，其中包含了每个顶点的直接邻居。
    # correct right-hemisphere triangles
    tris_lr_adjusted = deepcopy(tris_lr)
    # the right hemisphere indices start at zero, we need to offset them to start where left hemisphere indices end.
    tris_lr_adjusted[1] += int(numberOfDipoles/2)#调整右半脑的索引，以便从左半脑索引结束的地方开始
    # left and right hemisphere
    for hem in range(2):#遍历左右两个半球，对于每个顶点的索引：
        for idx in range(numberOfDipoles):#对每个顶点进行循环
            # Find the indices of the triangles where our current dipole idx is part of,找到包含当前顶点的所有三角形。将这些三角形的所有顶点添加到当前顶点的邻居列表中。
            trianglesOfIndex = tris_lr_adjusted[hem][np.where(tris_lr_adjusted[hem] == idx)[0], :]
            for tri in trianglesOfIndex:
                neighbors[idx].extend(tri)
                # Remove self-index (otherwise neighbors[idx] is its own neighbor)移除邻居列表中的当前顶点，以避免将自身包括在邻居中。
                neighbors[idx] = list(filter(lambda a: a != idx, neighbors[idx]))
            # Remove duplicates,去除邻居列表中的重复项。
            neighbors[idx] = list(np.unique(neighbors[idx]))
    #print("util get_triangle_neighbors") 
    return neighbors#返回包含每个顶点邻居索引的列表。


def calculate_source(data_obj, fwd, duration_of_trial=None, baseline_span=(-0.2, 0.0), 
    data_span=(0, 0.5), n_samples=int(1e4), optimizer=None, learning_rate=0.001, 
    validation_split=0.1, n_epochs=60, metrics=None, device=None, delta=1, 
    batch_size=8, loss=None, false_positive_penalty=2, parallel=False, 
    verbose=True):
    ''' The all-in-one convenience function for esinet.这个函数是一个方便的接口，用于使用esinet执行整个逆问题求解的流程
    
    Parameters
    ----------
    data_obj : mne.Epochs, mne.Evoked，`M/EEG 数据的 mne 对象，可以是 Epochs 或平均（Evoked）响应
        The mne object holding the M/EEG data. Can be Epochs or 
        averaged (Evoked) responses.
    fwd : mne.Forward， 正演模型。
        The forward model
    baseline_span : tuple，基线时间范围（单位：秒）。
        The time range of the baseline in seconds.
    data_span : tuple，数据时间范围（单位：秒）
        The time range of the data in seconds.
    n_samples : int，用于模拟的训练样本数量，越高逆问题的解越准确。
        The number of training samples to simulate. The 
        higher, the more accurate the inverse solution
    optimizer : tf.keras.optimizers， 用于反向传播的优化器。
        The optimizer that for backpropagation.
    learning_rate : float，用于训练神经网络的学习率。
        The learning rate for training the neural network
    validation_split : float， 保留作为验证集的数据比例。
        Proportion of data to keep as validation set.
    n_epochs : int，训练的轮数，在一个 epoch 中，所有训练样本都用于训练一次。
        Number of epochs to train. In one epoch all training samples 
        are used once for training.
    metrics : list/str，用于性能监控的指标。
        The metrics to be used for performance monitoring during training.
    device : str，要使用的设备，例如图形卡。
        The device to use, e.g. a graphics card.
    delta : float， 控制 Huber 损失。
        Controls the Huber loss.
    batch_size : int， 在反向传播期间同时计算错误的样本数量。
        The number of samples to simultaneously calculate the error 
        during backpropagation.
    loss : tf.keras.losses， 损失函数
        The loss function.
    false_positive_penalty : float，定义假阳性预测的权重。增加以获得保守的逆解，减少以获得自由的预测。
        Defines weighting of false-positive predictions. Increase for conservative 
        inverse solutions, decrease for liberal prediction.
    verbose : int/bool， 控制程序的详细程度。
        Controls verbosity of the program.
    
    
    Return
    ------
    source_estimate : mne.SourceEstimate，返回源估计对象。
        The source

    Example
    -------

    source_estimate = calculate_source(epochs, fwd)
    source_estimate.plot()

    '''
    # Calculate signal to noise ratio of the data， 计算数据的信噪比。
    target_snr = calc_snr_range(data_obj, baseline_span=baseline_span, data_span=data_span)
    if duration_of_trial is None:
        duration_of_trial = data_obj.times[-1] - data_obj.times[0]
    # Simulate sample M/EEG data，使用仿真类创建模拟器，用于生成样本 M/EEG 数据。
    sim = simulation.Simulation(fwd, data_obj.info, settings=dict(duration_of_trial=duration_of_trial, target_snr=target_snr), parallel=parallel, verbose=True)
    sim.simulate(n_samples=n_samples)#进行仿真以生成样本 M/EEG 数据。
    # Train neural network
    neural_net = net.Net(fwd, verbose=verbose)#创建神经网络模型。
    if int(verbose) > 0:# 如果 verbose 大于 0，则打印神经网络的摘要。
        neural_net.summary()
        
    neural_net.fit(sim, optimizer=optimizer, learning_rate=learning_rate, 
        validation_split=validation_split, epochs=n_epochs, metrics=metrics,
        device=device, delta=delta, batch_size=batch_size, loss=loss,
        false_positive_penalty=false_positive_penalty)#使用模拟数据对神经网络进行训练。
    # Predict sources，使用训练好的神经网络进行预测。
    source_estimate = neural_net.predict(data_obj)
    #print("util calculate_source") 
    return source_estimate#返回源估计对象。

def get_source_diam_from_order(order, pos, dists=None):
    ''' Calculate the estimated source diameter given the neighborhood order.
     Useful to calculate source extents using the region_growing method in the
     esinet.Simulation object.根据邻域顺序计算估计的源直径。用于通过 esinet.Simulation 对象中的 region_growing 方法计算源范围时很有用。

    Parameters
    ----------
    order : int, 感兴趣的邻域顺序
        The neighborhood order of interest
    pos : numpy.ndarray，源模型偶极子的位置。
        Positions of the source model dipoles.
   可选的 dists 参数用于预先计算偶极子之间的距离。
    '''
    # pos = unpack_fwd(fwd)[2]
    if dists is None:
        dists = cdist(pos, pos)#该函数使用 cdist 计算偶极子位置之间的成对距离。
    dists[dists==0] = np.nan#然后，将零距离替换为 NaN，以避免零距离引起的问题。
    #print("util get_source_diam_from_order") 
    return np.median(np.nanmin(dists, axis=0))*(2+order)#它计算每个偶极子的最小距离的中值，并乘以 (2 + order) 以估计源直径。

def get_eeg_from_source(stc, fwd, info, tmin=-0.2):
    ''' Get EEG from source by projecting source activity through the lead field.通过将源活动通过前向模型投影来从源获取 EEG。
    将源活动通过前向模型投影来生成模拟 EEG 数据。
    Parameters
    ----------
    stc : mne.SourceEstimate，包含源数据的 SourceEstimate 对象
        The source estimate object holding source data.
    fwd : mne.Forawrd，前向模型。
        The forward model.
    
    Return
    ------
    evoked : mne.EvokedArray，返回 evoked : mne.EvokedArray，EEG 数据对象。
        The EEG data oject.
    '''
    fwd = deepcopy(fwd)
    fwd = fwd.copy().pick_channels(info['ch_names'])#它使用 deepcopy 复制前向模型，并使用 info['ch_names'] 选择相关通道
    leadfield = fwd['sol']['data']#从前向模型中提取导联场。
    eeg_hat = np.matmul(leadfield, stc.data)#通过矩阵乘法 (np.matmul) 将源活动投影到导联场上。
    #print("util get_eeg_from_source") 
    return mne.EvokedArray(eeg_hat, info, tmin=tmin)#使用模拟的 EEG 数据创建一个 mne.EvokedArray 对象

def mne_inverse(fwd, epochs, method='eLORETA', snr=3.0, 
    baseline=(None, None), rank='info', weight_norm=None, 
    reduce_rank=False, inversion='matrix', pick_ori=None, 
    reg=0.05, regularize=False,verbose=True):
    ''' Quickly compute inverse solution using MNE methods，此函数使用 MNE 方法快速计算逆问题解决方案。
    '''
    #检查输入的 epochs 是 mne.Epochs 对象还是 mne.Evoked 对象。如果是 mne.Epochs 对象，则取其平均值，如果是 mne.Evoked 对象，则直接使用。
    if isinstance(epochs, (mne.Epochs, mne.EpochsArray)):
        evoked = epochs.average()
    elif isinstance(epochs, (mne.Evoked, mne.EvokedArray)):
        evoked = epochs
    if 'eeg' in set(evoked.get_channel_types()):#如果数据包含 EEG 信道，则对数据应用平均参考。
        evoked.set_eeg_reference(projection=True, verbose=verbose)
    
    raw = mne.io.RawArray(evoked.data, evoked.info, verbose=verbose)#将 evoked 数据转换为 mne.io.RawArray 对象。
    if rank is None:#如果未提供 rank，则使用 mne.compute_rank 计算。
        rank = mne.compute_rank(epochs, tol=1e-6, tol_kind='relative', verbose=verbose)
    estimator = 'empirical'#使用 'empirical' 方法计算噪声协方差矩阵。

    if not all([v is None for v in baseline]):#如果提供了 baseline，则使用它计算噪声协方差矩阵。
        print("calcing a good noise covariance", baseline)
        
        try:
            noise_cov = mne.compute_covariance(epochs, tmin=baseline[0], 
                tmax=baseline[1], method=estimator, rank=rank, verbose=verbose)
        except:
            noise_cov = mne.compute_raw_covariance(
                raw, tmin=abs(baseline[0]), tmax=baseline[1], rank=rank, method=estimator,
                verbose=verbose)
    else:
        # noise_cov = mne.make_ad_hoc_cov(evoked.info, std=dict(eeg=1),
        #     verbose=verbose)
        noise_cov = None
    if regularize and noise_cov is not None:#如果设置了正则化标志，并且存在噪声协方差矩阵，则进行正则化。
        noise_cov = mne.cov.regularize(noise_cov, epochs.info, rank=rank, 
            verbose=verbose)

    if method.lower()=='beamformer' or method.lower()=='lcmv' or method.lower()=='beamforming':
        if baseline[0] is None:
            tmin = 0
        else:
            tmin = abs(baseline[0])
        # print(raw, tmin, rank, estimator)
        try:
            data_cov = mne.compute_covariance(epochs, tmin=baseline[1], 
                tmax=None, method=estimator, verbose=verbose)

        except:
            data_cov = mne.compute_raw_covariance(raw, tmin=baseline[1], 
                tmax=None, rank=rank, method=estimator, verbose=verbose)
        
 
        if regularize:
            data_cov = mne.cov.regularize(data_cov, epochs.info, 
                rank=rank, verbose=verbose)



        
        lcmv_filter = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=reg, 
            weight_norm=weight_norm, noise_cov=noise_cov, verbose=verbose, pick_ori=pick_ori, 
            rank=rank, reduce_rank=reduce_rank, inversion=inversion)

        stc = mne.beamformer.apply_lcmv(evoked.crop(tmin=0.), lcmv_filter, 
            max_ori_out='signed', verbose=verbose)

        
    else:
        if noise_cov is None:
            noise_cov = mne.make_ad_hoc_cov(evoked.info, std=dict(eeg=1), verbose=verbose)
        # noise_cov = mne.cov.regularize(noise_cov, raw.info, verbose=verbose)
        lambda2 = 1. / snr ** 2
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            evoked.info, fwd, noise_cov, loose='auto', depth=None, fixed=True, 
            verbose=verbose)
            
        stc = mne.minimum_norm.apply_inverse(evoked.crop(tmin=0.), inverse_operator, lambda2,
                                    method=method, return_residual=False, verbose=verbose)
    #print("util mne_inverse") 
    return stc #返回计算得到的源时间序列

def wrap_mne_inverse(fwd, sim, method='eLORETA', snr=3.0, parallel=True, 
    add_baseline=False, n_baseline=200, 
    rank='info', reduce_rank=False, weight_norm=None, inversion='matrix', 
    pick_ori=None, reg=0.05, regularize=False,):
    ''' Wrapper that calculates inverse solutions to a bunch of simulated
        samples of a esinet.Simulation object，
        wrap_mne_inverse 函数是对 mne_inverse 函数的封装，用于计算 esinet 模拟对象的多个样本的逆解。
    '''
    eeg, sources = net.Net._handle_data_input((deepcopy(sim),))
    if add_baseline:
        eeg = [add_noise_baseline(e, src, fwd, num=n_baseline, verbose=0) for e, src in zip(eeg, sources)]
        baseline = (eeg[0].tmin, 0)
    else:
        baseline = (None, None)
    # print(eeg)
    n_samples = sim.n_samples
    
    if n_samples < 4:
        parallel = False
    
    if parallel:
        stcs = Parallel(n_jobs=-1, backend="loky") \
            (delayed(mne_inverse)(fwd, eeg[i], method=method, snr=snr, 
                baseline=baseline, rank=rank, weight_norm=weight_norm,
                reduce_rank=reduce_rank, inversion=inversion, 
                pick_ori=pick_ori, reg=reg, regularize=regularize) \
            for i in tqdm(range(n_samples)))
    else:
        stcs = []
    
        for i in tqdm(range(n_samples)):
            try:
                stc = mne_inverse(fwd, eeg[i], method=method, snr=snr, 
                    baseline=baseline, rank=rank, weight_norm=weight_norm,
                    reduce_rank=reduce_rank, inversion=inversion, 
                    pick_ori=pick_ori, reg=reg, regularize=regularize)
            except:
                print(f'{method} didnt work, returning zeros')
                stc = deepcopy(stcs[0])
                stc.data = np.zeros((stc.data.shape[0], len(eeg[i].crop(tmin=0.).times)))
                
                    
            stcs.append(stc)
    #print("util wrap_mne_inverse") 
    return stcs


def convert_simulation_temporal_to_single(sim):#函数的目的是将 esinet 模拟对象的时间维度转换为单个时间点，以便与不同形状的数据进行处理。
    sim_single = deepcopy(sim)
    sim_single.temporal = False
    sim_single.settings['duration_of_trial'] = 0

    eeg_data_lstm = sim.eeg_data.get_data()
    # Reshape EEG data
    eeg_data_single = np.expand_dims(np.vstack(np.swapaxes(eeg_data_lstm, 1,2)), axis=-1)
    # Pack into mne.EpochsArray object
    epochs_single = mne.EpochsArray(eeg_data_single, sim.eeg_data.info, 
        tmin=sim.eeg_data.tmin, verbose=0)
    
    # Reshape Source data
    source_data = np.vstack(np.swapaxes(np.stack(
        [source.data for source in sim.source_data], axis=0), 1,2)).T
    # Pack into mne.SourceEstimate object
    source_single = deepcopy(sim.source_data[0])
    source_single.data = source_data
    
    # Copy new shaped data into the Simulation object:
    sim_single.eeg_data = epochs_single
    sim_single.source_data = source_single
    #print("util convert_simulation_temporal_to_single")
    return sim_single

def collapse(x):
    ''' Collapse a  3D matrix (samples x dipoles x timepoints) into a 2D array 
    by combining the first and last dim.
    collapse 函数的目的是将一个三维矩阵（样本 x 偶极子 x 时间点）转换为一个二维数组，通过合并第一个和最后一个维度
    Parameters
    ----------
    x : numpy.ndarray
        Three-dimensional matrix, e.g. (samples x dipoles x timepoints)
    '''
    #print("collapse")
    return np.swapaxes(x, 1,2).reshape(int(x.shape[0]*x.shape[2]), x.shape[1])#一个二维数组，形状为 (样本数 * 时间点数, 偶极子数)。

def custom_logger(logger_name, level=logging.DEBUG):
    """
    Creates a new log file to log to.
    custom_logger 函数的目的是创建一个新的日志文件，用于记录日志信息。
    Parameters
    ----------
    logger_name : str
        Name or path of the logger
    level : see logging module for further information
    
    Return
    ------
    logger : logging.getLogger 
        Logger handle

    Example
    -------
    logger1 = custom_logger('path_to_logger/mylogfile')
    logger1.info('Here is some info written to the file')

    logger2 = custom_logger('path_to_logger/mylogfile2')
    logger2.info('Here is some different info written to the file')

    
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(levelname)s: "
                    "%(message)s")
    log_format = logging.Formatter(format_string)

    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    #print("custom_logger")
    return logger

def load_net(path, name='instance', custom_objects={}):#load_net 函数的目的是从磁盘加载已保存的 TensorFlow 模型和相关对象
    import tensorflow as tf
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    with open(path + f'\\{name}.pkl', 'rb') as f:
        net = pkl.load(f)
    net.model = model
    #print("load_net")
    return net

def create_n_dim_noise(shape, exponent=4):
    ''' Creates n-dimensional noise of given shape. The frequency spectrum is given 
    by the exponent, whereas exponent=2 gives pink noise, although exponent=4 looks more like it imo.
    create_n_dim_noise 函数的目的是生成指定形状的多维噪声，其频谱由给定的指数决定。这个函数生成的噪声在频谱上表现为粉噪声，其中指数的不同影响颜色。
    Parameters
    ----------
    shape : tuple，shape：元组，指定所需噪声的形状。
        Desired shape of the noise
    exponent : int/float，频谱的指数，影响颜色。默认为4。
        Frequency spectrum
    
    '''
    signal = np.random.uniform(-1, 1, shape)#生成形状为 shape 的均匀分布的随机信号。
    signal_fft = np.fft.fftn(signal, ) / shape[0]#对信号进行 n 维傅立叶变换，并归一化。

    freqs = [np.sqrt(np.arange(s) + 1) for s in shape]#生成频率数组，其中每个频率对应于每个维度。应用指定指数的粉噪声频谱。
    if len(shape) == 1:
        # pinked_fft = signal_fft / np.sqrt( (freqs[0]**exponent)[np.newaxis, :] )
        pinked_fft = signal_fft / (freqs[0]**exponent)[np.newaxis, :]
    elif len(shape) == 2:
        # pinked_fft = signal_fft / np.sqrt( ((freqs[0]**exponent)[np.newaxis, :]+(freqs[1]**exponent)[:, np.newaxis]) )
        pinked_fft = signal_fft / ((freqs[0]**exponent)[np.newaxis, :]+(freqs[1]**exponent)[:, np.newaxis])
    elif len(shape) == 3:
        # pinked_fft = signal_fft / np.sqrt( ((freqs[0]**exponent)[:, np.newaxis, np.newaxis]+(freqs[1]**exponent)[np.newaxis, :, np.newaxis]+(freqs[2]**exponent)[np.newaxis, np.newaxis, :]))
        pinked_fft = signal_fft /  ((freqs[0]**exponent)[:, np.newaxis, np.newaxis]+(freqs[1]**exponent)[np.newaxis, :, np.newaxis]+(freqs[2]**exponent)[np.newaxis, np.newaxis, :])
    elif len(shape) == 4:
        # pinked_fft = signal_fft / np.sqrt( ((freqs[0]**exponent)[:, np.newaxis, np.newaxis]+(freqs[1]**exponent)[np.newaxis, :, np.newaxis]+(freqs[2]**exponent)[np.newaxis, np.newaxis, :]))
        pinked_fft = signal_fft /  ((freqs[0]**exponent)[:, np.newaxis, np.newaxis, np.newaxis]+(freqs[1]**exponent)[np.newaxis, :, np.newaxis, np.newaxis]+(freqs[2]**exponent)[np.newaxis, np.newaxis, :, np.newaxis]+(freqs[3]**exponent)[np.newaxis, np.newaxis, np.newaxis, :])
    pink = np.fft.ifftn(pinked_fft).real#对频谱进行 n 维傅立叶逆变换，得到实部，生成粉噪声。
    #print("create_n_dim_noise")
    return pink#返回生成的粉噪声

def vol_to_src(neighbor_indices, src_3d, pos):
    '''Interpolate a 3D source to a irregular grid using k-nearest 
    neighbor interpolation.vol_to_src 函数的目的是在不规则网格上使用 k-最近邻插值法对三维源进行插值。
    '''
    src_3d_flat = src_3d.flatten()
    src = src_3d_flat[neighbor_indices].mean(axis=-1)
    #print("vol_to_src")
    return src 


def batch_nmse(y_true, y_pred):
    y_true = np.stack([y/np.abs(y).max() for y in y_true.T], axis=1)
    y_pred = np.stack([y/np.abs(y).max() for y in y_pred.T], axis=1)
    nmse = np.nanmean((y_true.flatten()-y_pred.flatten())**2)
    #print("batch_nmse")
    return nmse

def batch_corr(y_true, y_pred):
    # y_true = np.stack([y/np.abs(y).max() for y in y_true.T], axis=1)
    # y_pred = np.stack([y/np.abs(y).max() for y in y_pred.T], axis=1)
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    #print("batch_corr")
    return r

def add_noise_baseline(eeg, src, fwd, num=50, verbose=True):
    ''' Adds noise baseline segment to beginning of trial of specified length.

    Parameters
    ----------
    eeg : mne.Epochs
        The mne Epochs object containing a single EEG trial
    src : mne.SourceEstimate
        The mne SourceEstimate object corresponding to the eeg
    fwd: mne.Forward
        the mne Forward model
    num : int,
        Number of data points to add as baseline.
    verbose : None/ bool
        Controls verbosity of the function
    '''
    
    noisy_eeg = eeg.get_data()[0]
    true_eeg = np.matmul(unpack_fwd(fwd)[1], src.data)
    noise = noisy_eeg - true_eeg
    if num>=noise.shape[1]:
        multiplier = np.ceil(num / noise.shape[1]).astype(int)+1
        noise = np.repeat(noise, multiplier, axis=1)
    start_idx = np.random.choice(np.arange(0, noise.shape[1]-num))
    noise_piece = noise[:, start_idx:start_idx+num]
    new_eeg = np.append(noise_piece, deepcopy(noisy_eeg), axis=1)
    sr = eeg.info['sfreq']
    new_eeg = mne.EpochsArray(new_eeg[np.newaxis, :, :], eeg.info, tmin=-num/sr, verbose=verbose)
    # new_eeg.set_eeg_reference('average', projection=True, verbose=verbose)
    #print("add_noise_baseline")
    return new_eeg



def multipage(filename, figs=None, dpi=300, png=False):#multipage 函数的目的是将所有已打开的（或列表中的）图形保存为指定名称的 PDF 文件，并可选择是否同时保存为 PNG 格式。
    ''' Saves all open (or list of) figures to filename.pdf with dpi''' 
    pp = PdfPages(filename)
    path = os.path.dirname(filename)
    fn = os.path.basename(filename)[:-4]

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        print(f'saving fig {fig}\n')
        fig.savefig(pp, format='pdf', dpi=dpi)
        if png:
            fig.savefig(f'{path}\\{i}_{fn}.png', dpi=600)
    pp.close()
    #print("multipage")