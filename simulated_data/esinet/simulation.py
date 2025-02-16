from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pickle as pkl
import dill as pkl
import random
from joblib import Parallel, delayed
# from tqdm.notebook import tqdm
from tqdm import tqdm
from mne.channels.layout import _find_topomap_coords
import colorednoise as cn
import mne
from time import time
#import util
#from .util.util import *
from .util import util

DEFAULT_SETTINGS = {#定义一个默认设置字典，包含了模拟的一些默认参数，例如源的数量、大小、电流、形状、试验时长、采样频率、信噪比等等
    'method': 'standard',#模拟方法为标准法
    'number_of_sources': (1, 2),#源的数量范围
    'extents':  (5, 40),  # in millimeters源的直径范围，单位是mm
    'amplitudes': (3, 10),#源的电流大小范围，单位是nAm，也就是0.001—100nAm
    'shapes': 'mixed',#源的形状，可以是高斯分布或者均匀分布或者二者混合分布，随机选择 'gaussian' 或 'flat' 形状
    'duration_of_trial': 0.1,#试验时长，以秒为单位
    'sample_frequency': 250,#数据的采样频率
    'target_snr': 10,#目标信噪比的范围
    'beta': (0, 3), 
    'beta_noise': (0, 3),#噪声的频谱斜率范围
    'beta_source': (0, 3),#源的频谱斜率范围
    'source_spread': "region_growing",#源的空间分布，混合分布，为每个源随机选择源扩展类型，可以是 'region_growing' 或 'spherical'
    'source_number_weighting': True,#指示是否对源的数量进行加权,也就是从设定的范围内随机选择一个值作为源的数量
    'source_time_course': "random",#源的时间课程类型是随机
}
class Simulation:#定义Simulation类，用于模拟源数据和M/EEG数据
    ''' Simulate and hold source and M/EEG data.
    
    Attributes
    ----------
    settings : dict
        The Settings for the simulation. Keys:

        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two numbers 
            specifying a range.
        extents : int/float/tuple/list
            size of sources in mm. Can be a single number or a list of two 
            numbers specifying a range.
        amplitudes : int/float/tuple/list
            the current of the source in nAm
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' or 'flat' 
            (i.e. uniform) or 'mixed'.
        duration_of_trial : int/float
            specifies the duration of a trial.
        sample_frequency : int
            specifies the sample frequency of the data.
        target_snr : float/tuple/list
            The desired average SNR of the simulation(s)
        beta : float/tuple/list
            The desired frequency spectrum slope (1/f**beta) of the noise. 
    fwd : mne.Forward 类属性fwd，是一个Forward对象，用于存储前向模型的信息，用于将源活动映射到传感器空间
        The mne-python Forward object that contains the forward model
    source_data : mne.sourceEstimate  类属性，是一个sourceEstimate对象，用于存储源活动的数据，可能包括源的时间活动和空间位置等信息
        A source estimate object from mne-python which contains the source 
        data.
    eeg_data : mne.Epochs  类属性，是一个Epochs对象，用于存储eeg数据
        A mne.Epochs object which contains the EEG data.
    n_jobs : int   类属性，指定要使用的核心数量，并行处理时，可以使用多个核心以加速计算
        The number of jobs/cores to utilize.
    
    Methods
    -------
    simulate : Simulate source and EEG data 模拟源活动和eeg数据
    plot : plot a random sample source and EEG 绘制一个随机样本的源活动和eeg数据

    '''
    #以下是类Simulation的构造函数，负责在创建类的实例时进行初始化
    def __init__(self, fwd, info, settings=DEFAULT_SETTINGS, n_jobs=-1, 
        parallel=False, verbose=False):#接受多个参数，-1表示使用所有可用核心，不并行处理，不启用详细输出（该设置仅影响当前类的实例）
        self.settings = settings#赋给类的属性settings，以便后续可以访问和使用这些模拟设置
        self.source_data = None#初始化为none，表示此时还没有源数据
        self.eeg_data = None#表示此时还没有脑电数据
        self.fwd = deepcopy(fwd)#创建fwd的深拷贝，完全独立的复制fwd中的所有数据，且修改self.fwd，原始fwd不受影响
        #self.fwd.pick_channels(info['ch_names'])
        self.fwd=self.fwd.pick_channels(info['ch_names'])#使用前向模型的pick_channels方法，仅保留与输入信息info中通道名称匹配的通道，这有助于确保前向模型与输入数据匹配。
        self.check_info(deepcopy(info))#调用类内部的check_info方法，用于检查和处理信息对象info，创建了 info 的深拷贝，以确保不会修改原始信息数据。

        self.check_settings()#调用类内部的check_settings方法，用于检查和处理模拟设置。
        self.settings['sample_frequency'] = info['sfreq']# 将信息对象info中的采样频率sfreq赋给模拟设置的sample_frequency键。
        self.info=info
        # self.info['sfreq'] = self.settings['sample_frequency']
        self.prepare_simulation_info()#调用类内部的 prepare_simulation_info 方法，用于准备模拟信息。这可能涉及计算一些额外的模拟参数或信息。
        self.subject = self.fwd['src'][0]['subject_his_id']#从前向模型的属性中获取源的主题标识（subject_his_id），并将其赋给类的属性subject。
        self.n_jobs = n_jobs#将构造函数中传入的 n_jobs 参数赋给类的属性 n_jobs，以指定用于并行处理的作业/核心数量。
        self.parallel = parallel#将构造函数中传入的 parallel 参数赋给类的属性 parallel，以标识是否要并行处理。
        self.verbose = verbose#将构造函数中传入的 verbose 参数赋给类的属性 verbose，以标识是否启用详细输出。
        self.diams = None#将类的属性 diams 初始化为 None，表示在创建类的实例时还没有直径数据
        
    
    def __add__(self, other):#用于定义两个类对象相加时的行为
        new_object = deepcopy(self)#创建了一个类的深拷贝，即 self 的副本，以确保不会修改原始对象。
        new_object.source_data.extend(other.source_data)#将self对象中的source_data与other对象中的source_data进行合并，以扩展 new_object 的 source_data
        new_object.eeg_data.extend(other.eeg_data)#将 self 对象中的 eeg_data 与 other 对象中的 eeg_data 进行合并，以扩展 new_object 的 eeg_data。
        new_object.simulation_info = pd.concat([
            new_object.simulation_info, 
            other.simulation_info
            ])#将 self 对象中的 simulation_info 与 other 对象中的 simulation_info 进行合并，以创建一个新的 simulation_info，这是一个包含模拟信息的 Pandas DataFrame。
        #  Deprecated
        # new_object.simulation_info.append(other.simulation_info)
        new_object.n_samples += other.n_samples#将 self 对象中的 n_samples 与 other 对象中的 n_samples 相加，以更新 new_object 的 n_samples。
        if new_object.settings["method"] != other.settings["method"]:#检查 self 和 other 对象的 settings 中的方法是否不同
            new_object.settings["method"] = "mixed"#如果 self 和 other 对象的方法不同，将 new_object 的方法设置为 "mixed"。
        return new_object#返回创建的新对象 new_object，这是两个原始对象的合并
        
    #def check_info(self, info):
        #self.info = info.pick_channels(self.fwd.ch_names, ordered=True)
    #def check_info(self, info):
        #picked_info = info.pick(self.fwd.info['ch_names'])
        #return picked_info
    #def check_info(self, info):
        #picked_info = info.pick_channels(self.fwd.info['ch_names'], ordered=True)
        #return picked_info
    def check_info(self, info):#该方法用于检查和设置类的 info 属性，以匹配前向模型中包含的通道
    # Get the channels present in the forward model
        fwd_ch_names = self.fwd['info']['ch_names']
    # Pick channels from the info that match the forward model
        #self.info = info.pick_channels(fwd_ch_names, ordered=True)#使用info下的pick_channels方法，从输入的 info 对象中挑选出与前向模型匹配的通道，以更新类的 info 属性。ordered=True 参数表示保持通道的顺序
        #self.info = mne.pick_channels(info['ch_names'], fwd_ch_names, ordered=True)#使用mne.pick_channels函数，用前向模型的通道列表 (fwd_ch_names) 来筛选和更新 info 对象的通道，以确保匹配前向模型的通道设置
        self.info = mne.pick_channels(info['ch_names'], include=fwd_ch_names, ordered=True)
        
    def prepare_simulation_info(self):#该方法用于准备一个空的 Pandas DataFrame，该DataFrame 用于存储模拟信息。
        self.simulation_info = pd.DataFrame(columns=['number_of_sources', 'positions', 'extents', 'amplitudes', 'shapes', 'target_snr', 'betas', 'betas_noise', 'duration_of_trials', 'beta_source'])
    def simulate(self, n_samples=500):#该方法用于模拟源和 EEG 数据，根据类的设置参数生成模拟数据
        ''' Simulate sources and EEG data 这里的10000是默认的样本数，是一个合理的默认值'''
        self.n_samples = n_samples
        self.source_data = self.simulate_sources(n_samples)
        self.eeg_data = self.simulate_eeg()
        return self
    
    
 
    def plot(self):#plot 方法在此处没有提供具体实现，可能是用于可视化的占位符方法
        pass
    
    def simulate_sources(self, n_samples):#这是模拟源数据的核心方法，根据不同的模拟方法来生成模拟数据，这个方法将返回一个包含模拟的源数据的 mne.SourceEstimate 对象。模拟方法和并行处理方式取决于 self.settings["method"] 和 self.parallel 的设置。这个方法的目的是生成与模拟相关的源数据。
        
        n_dip = self.pos.shape[0]#获取源的数量，pos 存储了源的位置信息。
        # source_data = np.zeros((n_samples, n_dip, n_time), dtype=np.float32)
        source_data = []#初始化一个空列表，用于存储生成的源数据。
        if self.verbose:#检查 verbose 参数是否为 True，如果是，就打印以下信息
            print(f'Simulate Source')


        if self.settings["method"] == "standard":#检查模拟方法是否为标准方法，这就是稀疏源方法
            print("Simulating data based on sparse patches.")
            if self.parallel:#如果启用了并行处理 (self.parallel 为 True)，则使用 joblib 库的并行处理功能来生成源数据。n_jobs 参数表示要使用的 CPU 核心数，backend 参数指定并行处理的后端。它将调用 simulate_source 方法来生成每个样本的源数据。
                source_data = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source)() 
                    for _ in tqdm(range(n_samples)))
            else:#如果不使用并行处理，使用循环生成源数据。
                for i in tqdm(range(n_samples)):
                    source_data.append( self.simulate_source() )
            
        elif self.settings["method"] == "noise":#检查模拟方法是否为噪声方法。在这个方法中，数据将基于 1/f 噪声生成。
            print("Simulating data based on 1/f noise.")
            self.prepare_grid()
            if self.parallel:#如果启用了并行处理，使用并行处理生成源数据。它将调用 simulate_source_noise 方法来生成每个样本的源数据。
                source_data = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source_noise)()
                    for _ in tqdm(range(n_samples)))
            else:#如果不使用并行处理，使用循环生成源数据。
                for i in tqdm(range(n_samples)):
                    source_data.append( self.simulate_source_noise() )
        elif self.settings["method"] == "mixed":#检查模拟方法是否为混合方法。在这个方法中，数据将同时基于 1/f 噪声和稀疏源模式生成。
            print("Simulating data based on 1/f noise and sparse patches.")
            self.prepare_grid()
            if self.parallel:#如果启用了并行处理，使用并行处理生成 1/f 噪声模拟的源数据（前半部分样本）
                source_data_tmp = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source_noise)() 
                    for _ in tqdm(range(int(n_samples/2))))
                for single_source in source_data_tmp:#将 1/f 噪声模拟的源数据添加到 source_data 列表中。
                    source_data.append( single_source )
                source_data_tmp = Parallel(n_jobs=self.n_jobs, backend='loky') \
                    (delayed(self.simulate_source)() #如果启用了并行处理，使用并行处理生成稀疏源模式模拟的源数据（后半部分样本）
                    for _ in tqdm(range(int(n_samples/2), n_samples)))

                for single_source in source_data_tmp:#将稀疏源模式模拟的源数据添加到 source_data 列表中。
                    source_data.append( single_source )
            else:
                for i in tqdm(range(int(n_samples/2))):
                    source_data.append( self.simulate_source_noise() )
                for i in tqdm(range(int(n_samples/2), n_samples)):
                    source_data.append( self.simulate_source() )
                    

        # Convert to mne.SourceEstimate 将生成的源数据转换为mne.SourceEstimate对象。这是一个必要的步骤，因为mne.SourceEstimate对象是MNE-Python库中用于表示源活动的标准对象，它包含了有关源活动的重要信息，如时域信号和空间分布。
         # Convert to mne.SourceEstimate
        if self.verbose:
            print(f'Converting Source Data to mne.SourceEstimate object')
        # if self.settings['duration_of_trial'] == 0:
        #     sources = util.source_to_sourceEstimate(source_data, self.fwd, 
        #         sfreq=self.settings['sample_frequency'], subject=self.subject) 
        # else:
        sources = self.sources_to_sourceEstimates(source_data)
        return sources

    def prepare_grid(self):#目的是生成用于模拟源数据的网格，并将网格的相关信息存储在 `self.grid` 中，以便后续使用。网格将用于模拟不同的源模式
        n = 10#表示网格的维度。
        n_time = np.clip(int(self.info['sfreq'] * np.max(self.settings['duration_of_trial'])), a_min=1, a_max=np.inf).astype(int)#计算 `n_time`，它表示模拟的时间点数量，根据模拟的持续时间和采样频率计算。
        shape = (n,n,n,n_time)#创建一个形状为(n, n, n, n_time)的数组shape，用于表示网格的形状。
        
        x = np.linspace(self.pos[:, 0].min(), self.pos[:, 0].max(), num=shape[0])#生成x轴的坐标，该坐标范围从源位置self.pos中x坐标的最小值到最大值，共shape[0]个点
        y = np.linspace(self.pos[:, 1].min(), self.pos[:, 1].max(), num=shape[1])#shape[1]个点
        z = np.linspace(self.pos[:, 2].min(), self.pos[:, 2].max(), num=shape[2])#shape[2]个点
        k_neighbors = 5#表示每个源周围的邻居数量
        grid = np.stack(np.meshgrid(x,y,z, indexing='ij'), axis=0)#生成一个三维网格grid，使用np.meshgrid函数，它将xyz坐标的组合作为输入，使用索引坐标。
        grid_flat = grid.reshape(grid.shape[0], np.product(grid.shape[1:])).T#将三维网格grid变换为一个二维数组grid_flat，其中每行代表一个网格点。
        neighbor_indices = np.stack([#计算每个源周围的邻居索引，对于每个源位置，找到距离最近的k_neighbors个网格点，并将其索引存储在neighbor_indices中
            np.argsort(np.sqrt(np.sum((grid_flat - coords)**2, axis=1)))[:k_neighbors] for coords in self.pos
        ], axis=0)

        self.grid = {#将生成的网格信息存储在self.grid字典中，包括网格的形状、邻居数量、指数、坐标等信息。
            "shape": shape,
            "k_neighbors": k_neighbors,
            "exponent": self.settings["beta_source"],
            "x": x,
            "y": y,
            "z": z,
            "grid": grid,
            "grid_flat": grid_flat,
            "neighbor_indices": neighbor_indices
        }
        
    def get_grid_info(self):
        if hasattr(self, 'grid'):
            return self.grid
        else:
            return "Grid information is not available."
    
    def simulate_source_noise(self):#生成源噪声数据
        exponent = self.get_from_range(self.grid["exponent"], dtype=float)#从指定范围中获取exponent值，用于模拟源数据的指数（即 1/f 的指数）。这决定了源数据的频谱特性
        src_3d = util.create_n_dim_noise(self.grid["shape"], exponent=exponent)#生成多维噪声数据src_3d，使用exponent指定的指数值。这个数据表示在空间和时间上的噪声分布
        duration_of_trial = self.get_from_range(#从模拟设置中获取模拟持续时间
            self.settings['duration_of_trial'], dtype=float)
        n_time = np.clip(int(round(duration_of_trial * self.info['sfreq'])), 1, None)#根据模拟持续时间和采样频率计算时间点数量 n_time。使用 np.clip函数将其限制在最小值为 1 以上
        if len(src_3d.shape) == 3:#检查 `src_3d` 的形状是否为三维。
            src_3d = src_3d[:,:,:,np.newaxis]
        src = np.zeros((self.pos.shape[0], n_time))#创建一个用于存储源数据的数组 `src`，形状为 (源数量, 时间点数量)
        for i in range(n_time):#循环遍历时间点
            #调用 `util.vol_to_src` 函数，将 `src_3d` 中的三维噪声数据映射到源空间，并将结果存储在 `src` 中。
            src[:, i] = util.vol_to_src(self.grid["neighbor_indices"], src_3d[:, :, :, i], self.pos)
        #创建一个字典 `d`，包含用于存储模拟信息的各项。
        d = dict(number_of_sources=np.nan, positions=[np.nan], extents=[np.nan], amplitudes=[np.nan], shapes=[np.nan], target_snr=0, duration_of_trials=duration_of_trial, beta_source=exponent)
        df_new = pd.DataFrame(columns=self.simulation_info.columns)#创建一个新的空数据框 `df_new`，其列名与 `self.simulation_info` 中的列名相同。
        self.simulation_info = pd.concat([self.simulation_info, df_new], ignore_index=True)
        for key, val in d.items():#遍历字典 `d` 的键和值。
            df_new.loc[0, key] = val#将 `d` 中的每个键值对存储在 `df_new` 数据框的第一行中
        df_new.reset_index(drop=True)#重置 `df_new` 的索引，将索引值按顺序排列。
        #将新的模拟信息数据框 `df_new` 连接到 `self.simulation_info` 中，以扩展模拟信息。
        self.simulation_info = pd.concat([self.simulation_info, df_new])
        # deprecated soon:
        # self.simulation_info = self.simulation_info.append(d, ignore_index=True)
        return src#返回生成的源数据 `src`
        
    def sources_to_sourceEstimates(self, source_data):#将源数据转化为 mne.SourceEstimate 对象。将其转换为源估计对象。
        template = util.source_to_sourceEstimate(source_data[0], 
                    self.fwd, sfreq=self.settings['sample_frequency'], 
        #使用 `source_to_sourceEstimate` 函数创建一个模板 `template`，其中包括模拟的第一个源数据
                    subject=self.subject)
        sources = []#创建一个空列表 `sources` 用于存储源估计对象。
        for source in tqdm(source_data):#循环遍历源数据列表，其中每个源对应于一个时间点
            tmp = deepcopy(template)#创建模板 `template` 的深拷贝 `tmp`
            tmp.data = source#将 tmp 的数据属性设置为当前源数据。
            sources.append(tmp)#将源估计对象 tmp 添加到 sources 列表中。
        return sources#返回包含源估计对象的列表 sources。


    def simulate_source(self):#成模拟的脑源信号数据。该方法基于模拟设置以及源的位置、数量、形状、幅度等参数来生成脑源信号。主要功能是生成源信号，模拟了不同形状和特性的源，可以用于后续的 EEG 数据模拟和分析。
        ''' Returns a vector containing the dipole currents. 说明该函数的主要目的是生成包含偶极子电流的向量
        Requires only a dipole position list and the simulation settings.表明该函数只需要两个输入，即偶极子的位置列表和模拟设置

        Parameters
        ----------
        pos : numpy.ndarray    
            (n_dipoles x 3), list of dipole positions.
       描述了参数 `pos`，它是一个 Numpy 数组，包含偶极子的位置信息。具体要求是 `(n_dipoles x 3)`，表示它应该是一个包含偶极子位置的 3D 数组，其中 `n_dipoles` 表示偶极子的数量。
        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two numbers specifying a range.
       描述了参数 `number_of_sources`，它表示生成的源的数量。可以是一个整数，也可以是一个包含两个数字的元组或列表，指定源的数量的范围。
        extents : int/float/tuple/list
            diameter of sources (in mm). Can be a single number or a list of 
            two numbers specifying a range.
       它表示源的直径（以毫米为单位）。可以是一个具体的数值，也可以是一个包含两个数字的元组或列表，指定直径的范围。
        amplitudes : int/float/tuple/list
            the current of the source in nAm
       描述了参数 `amplitudes`，它表示每个源的电流值，单位为 nAm（纳安培）。
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' or 'flat' 
            (i.e. uniform) or 'mixed'.
       描述了参数 `shapes`，它表示源信号的空间分布。可以是 'gaussian'（高斯）、'flat'（均匀）或 'mixed'（混合）
        duration_of_trial : int/float
            specifies the duration of a trial.
       描述了参数 `duration_of_trial`，它表示模拟的试验持续时间，以整数或浮点数表示。
        sample_frequency : int
            specifies the sample frequency of the data.
       描述了参数 `sample_frequency`，它表示数据的采样频率
        Return
        ------
        source : numpy.ndarray, (n_dipoles x n_timepoints), the simulated 
            source signal
       描述了返回的主要结果 `source`，它是一个 Numpy 数组，包含模拟的源信号数据。该数组的维度是 `(n_dipoles x n_timepoints)`，其中 `n_timepoints` 表示时间点的数量。
        simSettings : dict, specifications about the source.
       描述了返回值中的 `simSettings`，它是一个字典，包含了有关生成源的模拟设置的详细信息。
        Grova, C., Daunizeau, J., Lina, J. M., Bénar, C. G., Benali, H., & 
            Gotman, J. (2006). Evaluation of EEG localization methods using 
            realistic simulations of interictal spikes. Neuroimage, 29(3), 
            734-753.
        '''
        
        ###########################################
        # Select ranges and prepare some variables
        # Get number of sources:
        #根据模拟设置确定源的数量 number_of_sources。如果没有启用源数量的加权或者 settings["number_of_sources"] 是浮点数或整数，则直接使用 get_from_range 函数获取该数量。
        if not self.settings["source_number_weighting"] or isinstance(self.settings["number_of_sources"], (float, int)):
            number_of_sources = self.get_from_range(
                self.settings['number_of_sources'], dtype=int)
        else:
            population = np.arange(*self.settings["number_of_sources"])#如果启用了源数量的加权 (source_number_weighting)，则创建一个包含一系列值的数组 population，并为每个值分配相应的权重（逆值）。最后，从这个加权的分布中随机选择一个值，作为源的数量。
            weights = 1 / population
            weights /= weights.sum()
            number_of_sources = random.choices(population=population,weights=weights, k=1)[0]
        #如果启用了源扩展类型 source_spread 为 'mixed'，则为每个源随机选择源扩展类型，可以是 'region_growing' 或 'spherical'。否则，所有源都使用相同的扩展类型。
        if self.settings["source_spread"] == 'mixed':
            source_spreads = [np.random.choice(['region_growing', 'spherical']) for _ in range(number_of_sources)]
        else:
            source_spreads = [self.settings["source_spread"] for _ in range(number_of_sources)]

   
        extents = [self.get_from_range(self.settings['extents'], dtype=float) #使用 get_from_range 获取每个源的扩展直径 extents，这个值是浮点数，表示源的直径，单位为毫米。
            for _ in range(number_of_sources)]
        
        # Decide shape of sources根据模拟设置中的 shapes 决定每个源的形状。如果 shapes 为 'mixed'，则随机选择 'gaussian' 或 'flat' 形状，以列表的形式表示。如果 shapes 为 'gaussian' 或 'flat'，则所有源都使用相同的形状。
        if self.settings['shapes'] == 'mixed':
            shapes = ['gaussian', 'flat']*number_of_sources
            np.random.shuffle(shapes)
            shapes = shapes[:number_of_sources]
            if type(shapes) == str:
                shapes = [shapes]

        elif self.settings['shapes'] == 'gaussian' or self.settings['shapes'] == 'flat':
            shapes = [self.settings['shapes']] * number_of_sources
        
        # Get amplitude gain for each source (amplitudes come in nAm)获取每个源的电流值 amplitudes，根据模拟设置中的 amplitudes 来获取，并将其转换为标准单位 nAm（纳安培）。
        amplitudes = [self.get_from_range(self.settings['amplitudes'], dtype=float) * 1e-9 for _ in range(number_of_sources)]
        #随机选择源的中心位置 src_centers，这是在偶极子位置中选择的源数量，确保没有重复。通过从偶极子位置中选择源的数量的不重复索引，随机选择源的中心位置
        src_centers = np.random.choice(np.arange(self.pos.shape[0]), \
            number_of_sources, replace=False)
        #获取模拟设置中的试验持续时间 duration_of_trial
        duration_of_trial = self.get_from_range(
            self.settings['duration_of_trial'], dtype=float
        )
        #计算信号长度 signal_length，这是根据采样频率和试验持续时间计算得到的。根据采样频率 (sample_frequency) 和试验持续时间 (duration_of_trial) 计算信号长度，以确定信号的采样点数
        signal_length = int(round(self.settings['sample_frequency']*duration_of_trial))
        betas = []#创建一个空的列表 betas，用于存储每个源的 beta 值
        #如果信号长度大于 1，说明需要生成具有时间动态的信号。根据模拟设置 source_time_course 决定信号的生成方式。如果是 "pulse"，则生成脉冲信号；否则，生成根据 beta 参数的幂律功率谱高斯噪声信号。生成的信号被归一化，然后存储在 signals 列表中，同时将对应的 beta 值存储在 betas 列表中。根据模拟设置中的 source_time_course 决定信号的生成方式。如果是 "pulse"，则生成脉冲信号；否则，生成根据 beta 参数的幂律功率谱高斯噪声信号。
        if signal_length > 1:
            signals = []
            
            if self.settings["source_time_course"].lower() == "pulse":
                signals = [self.get_biphasic_pulse(signal_length) for _ in range(number_of_sources)]

            else:
                
                for _ in range(number_of_sources):
                    beta = self.get_from_range(self.settings['beta'], dtype=float)
                    signal = cn.powerlaw_psd_gaussian(beta, signal_length) 
                    signal /= np.max(np.abs(signal))
                    signals.append(signal)
                    betas.append(beta)
                    
            
            sample_frequency = self.settings['sample_frequency']
        else:  # else its a single instance
            sample_frequency = 0
            signal_length = 1
            signals = list(np.random.choice([-1, 1], number_of_sources))
        
        
        # sourceMask = np.zeros((self.pos.shape[0]))
        source = np.zeros((self.pos.shape[0], signal_length))#创建一个与偶极子位置一样大小的全零矩阵 source，该矩阵用于存储最终的源信号。
        ##############################################
        # Loop through source centers (i.e. seeds of source positions)遍历每个源的中心位置和相关参数，生成源信号。根据 source_spread 类型，如果是 "region_growing"，则计算与中心位置的距离，并获取相关的偶极子索引。如果是其他类型，直接获取预先计算好的距离矩阵。
        for i, (src_center, shape, amplitude, signal, source_spread) in enumerate(zip(src_centers, shapes, amplitudes, signals, source_spreads)):
            if source_spread == "region_growing":
                #使用 get_n_order_indices 函数获取相应订单 (order) 的索引 (d),计算所有源点到当前源 (src_center) 在订单索引范围内的距离，并存储在 dists 中。
                order = self.extents_to_orders(extents[i])
                d = np.array(get_n_order_indices(order, src_center, self.neighbors))
                # if isinstance(d, (int, float, np.int32)):
                #     d = [d,]
                dists = np.empty((self.pos.shape[0]))
                dists[:] = np.inf
                dists[d] = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))[d]
            else:
                # dists = np.sqrt(np.sum((self.pos - self.pos[src_center, :])**2, axis=1))
                #直接从预先计算好的 distance_matrix 中获取源点到当前源 (src_center) 的距离，并存储在 dists 中。
                dists = self.distance_matrix[src_center]
                d = np.where(dists<extents[i]/2)[0]
        #根据源的形状（'gaussian' 或 'flat'）生成源信号。如果形状是 'gaussian'，则创建一个高斯形状的源信号。如果形状是 'flat'，则创建一个均匀分布的源信号。如果源的形状是 "gaussian"，则根据源的影响区域（dists）的数量来确定源信号。如果 len(d) < 2，表示源的影响区域内只有一个点，将源信号初始化为零数组，然后将振幅 (amplitude) 赋值给该点。如果 len(d) >= 2，表示源的影响区域内有多个点，计算标准差 (sd)，然后根据高斯分布函数计算源信号，并乘以振幅和输入信号 (signal)。
            if shape == 'gaussian':
                
                if len(d) < 2:                    
                    activity = np.zeros((len(dists), 1))
                    activity[d, 0] = amplitude
            
                    
                    activity = activity * signal
                else:
                    sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                    activity = np.expand_dims(util.gaussian(dists, 0, sd) * amplitude, axis=1) * signal
                source += activity
            elif shape == 'flat':
                if not isinstance(signal, (list, np.ndarray)):
                    signal = np.array([signal])
                    activity = util.repeat_newcol(amplitude * signal, len(d))
                    if len(activity.shape) == 0:
                        activity = np.array([activity]).T[:, np.newaxis]
                    else:
                        activity = activity.T[:, np.newaxis]
                    
                else:
                    activity = util.repeat_newcol(amplitude * signal, len(d)).T
                if len(activity.shape) == 1:
                    if len(d) == 1:
                        activity = np.expand_dims(activity, axis=0)    
                    else:
                        activity = np.expand_dims(activity, axis=1)
                source[d, :] += activity 
            else:
                msg = BaseException("shape must be of type >string< and be either >gaussian< or >flat<.")
                raise(msg)
        
        # Document the sample创建字典 d，包含了生成源的相关信息，如源的数量、位置、直径、电流、形状、信噪比、试验持续时间和 beta 值。
        # Document the sample
        d = dict(number_of_sources=number_of_sources, positions=self.pos[src_centers], extents=extents, amplitudes=amplitudes, shapes=shapes, target_snr=0, duration_of_trials=duration_of_trial, betas=betas)
        df_new = pd.DataFrame(columns=self.simulation_info.columns)
        for key, val in d.items():
            df_new.loc[0, key] = val
        df_new.reset_index(drop=True)

        self.simulation_info = pd.concat([self.simulation_info, df_new])
        
        # deprecated soon:
        # self.simulation_info = self.simulation_info.append(d, ignore_index=True)
        #print("simulate_source")
        return source#返回生成的源信号 source

    def simulate_eeg(self):#用于模拟生成 EEG 数据,将源信号转换为 EEG 试验数据，并为每个试验添加噪声。最终，它将生成 mne.Epochs 对象，该对象包含了带有噪声的 EEG 数据，可以用于进一步的分析。这些数据将用于模拟 EEG 测量。
        ''' Create EEG of specified number of trials based on sources and some SNR.根据源和信噪比创建指定试验数量的EEG数据
        Parameters，参数
        -----------
        sourceEstimates : list ，一个包含 `mne.SourceEstimate` 对象的列表，表示用于生成 EEG 数据的源估计
                        list containing mne.SourceEstimate objects
        fwd : mne.Forward，`mne.Forward` 对象，表示前向模型，描述了从源到电极的映射关系
            the mne.Forward object
        target_snr : tuple/list/float, 一个可以是单个浮点数、浮点数元组/列表或两个浮点数的范围，用于指定所期望的信噪比（目标信噪比）
                    desired signal to noise ratio. Can be a list or tuple of two 
                    floats specifying a range.
        beta : float，一个浮点数，确定添加到信号的噪声的频谱，其关系为功率 = 1/f^beta。0 表示白噪声，1 表示粉噪声（1/f 频谱）
            determines the frequency spectrum of the noise added to the signal: 
            power = 1/f^beta. 
            0 will yield white noise, 1 will yield pink noise (1/f spectrum)
        n_jobs : int，一个整数，表示要并行运行的作业数量。如果为 -1，则使用所有可用的核心
                Number of jobs to run in parallel. -1 will utilize all cores.
        return_raw_data : bool，一个布尔值，如果为 True，则函数返回 `mne.SourceEstimate` 对象的列表，否则返回原始 EEG 数据
                        if True the function returns a list of mne.SourceEstimate 
                        objects, otherwise it returns raw data

        Return
        -------
        epochs : list，一个列表，可以包含 `mne.Epochs` 对象或原始 EEG 数据，具体返回值取决于 `return_raw_data` 参数的设置
                list of either mne.Epochs objects or list of raw EEG data 
                (see argument <return_raw_data> to change output)
        '''

        ##以下这段注释的代码的作用是：目的是为了获得用于创建 EEG 数据的源信号 `sources`，并确保 `sources` 具有正确的维度。具体的处理方式取决于 `self.source_data` 的类型，以确保最终的源信号具有正确的形状（期望的维度）。这段代码的最终目标是准备好源信号供后续的 EEG 数据生成使用。
        # Desired Dim of sources: (samples x dipoles x time points)#源信号的期望维度：（样本数 x 偶极子数 x 时间点数）
        # unpack numpy array of source data#解压缩源数据的 NumPy 数组
        # print(type(self.source_data))#打印 `self.source_data` 变量的数据类型
        # if isinstance(self.source_data, (list, tuple)):#检查 `self.source_data` 是否是 Python 中的列表或元组。如果是，将执行以下代码块。这里检查源数据的类型，因为它可以以多种不同的方式表示
        #     sources = np.stack([source.data for source in self.source_data], axis=0)#从 `self.source_data` 中提取源数据，并将它们叠加成一个 NumPy 数组 `sources`。`source.data` 用于获取 `self.source_data` 中每个源的数据。`np.stack` 用于将这些数据在新的维度（`axis=0`）上叠加在一起。最终，`sources` 将成为一个三维数组，其中的维度表示样本、偶极子和时间点
        # else:#如果源数据不是列表或元组，将执行以下代码块。

        #     sources = self.source_data.data.T#将self.source_data直接分配给sources，但首先使用 `.T` 属性对源数据进行转置操作。这是因为在某些情况下，源数据的维度可能需要进行转置，以满足期望的维度要求。此操作将确保sources具有正确的维度

        # if there is no temporal dimension...该部分首先处理源数据的维度，确保每个源都有时间维度。如果源数据中没有时间维度，会为每个源添加一个时间维度。
        for i, source in enumerate(self.source_data):#遍历self.source_data中的每个源数据。enumerate用于同时获取源数据和它们的索引。
            if len(source.shape) == 1:#检查当前源数据的形状是否为一维，即是否缺少时间维度
                self.source_data[i] = np.expand_dims(source, axis=-1)#如果源数据缺少时间维度（一维），则使用np.expand_dims在其末尾添加一个维度，以确保每个源数据都具有形状(dipoles, time points)
        print('source data shape: ', self.source_data[0].shape, self.source_data[1].shape)#用于在控制台上打印第一个和第二个源数据的形状。
                

        # Load some forward model objects
        n_elec = self.leadfield.shape[0]#获取电极（或传感器）的数量，即前向模型中的电极数目
        n_samples = np.clip(len(self.source_data), a_min=1, a_max=np.inf).astype(int)#确定了源数据中的样本数,首先计算self.source_data的长度，然后使用np.clip函数确保样本数至少为 1，最后将结果转换为整数数据类型

        target_snrs = [self.get_from_range(self.settings['target_snr'], dtype=float) for _ in range(n_samples)]#生成一个包含样本数目的 `target_snr` 值的列表。每个样本都可以有不同的信噪比（SNR），这些值从配置参数中获得
        betas_noise = [self.get_from_range(self.settings['beta_noise'], dtype=float) for _ in range(n_samples)]#这一行代码生成一个包含样本数目的 `beta_noise` 值的列表。`beta_noise` 用于确定加入信号的噪声的频谱特性

        # Document snr and beta into the simulation info
        self.simulation_info['betas_noise'] = betas_noise#将 `betas_noise` 值存储到模拟信息中，以便记录这些信息
        self.simulation_info['target_snr'] = target_snrs#将 `target_snr` 值存储到模拟信息中，以便记录这些信息
    
        # Desired Dim for eeg_clean: (samples, electrodes, time points)
        if self.verbose:#检查是否应打印详细信息
            print(f'\nProject sources to EEG...')#指示正在将源数据投影到 EEG 数据中。
        eeg_clean = self.project_sources(self.source_data)#使用self.source_data投影源信号到EEG数据。投影过程将源信号转换为 EEG 数据，以便进行后续的处理和添加噪声。
        print(type(eeg_clean), eeg_clean[0].shape)
        if self.verbose:
            print(f'\nCreate EEG trials with noise...')#指示正在创建带有噪声的 EEG 试验
        
        # Add noise
        self.noise_generator = NoiseGenerator(self.info)#创建了一个噪声生成器 NoiseGenerator，该生成器用于为 EEG 数据添加噪声。

        eeg_trials_noisy = []#创建一个空列表，用于存储带有噪声的 EEG 试验数据
        for sample in tqdm(range(n_samples)):#遍历样本数目的范围
            #在每个循环迭代中，调用 self.create_eeg_helper 方法，将 eeg_clean 中的 EEG 数据与相应的 SNR 和噪声特性组合在一起，然后将结果添加到 eeg_trials_noisy 列表中。
            eeg_trials_noisy.append( self.create_eeg_helper(eeg_clean[sample], 
                target_snrs[sample], betas_noise[sample]) 
            )

        for i, eeg_trial_noisy in enumerate(eeg_trials_noisy):#遍历 eeg_trials_noisy 列表中的每个 EEG 试验。
            if len(eeg_trial_noisy.shape) == 2:#检查当前 EEG 试验是否具有二维形状
                #print("expanding")
                eeg_trials_noisy[i] = np.expand_dims(eeg_trial_noisy, axis=0)#如果 EEG 试验是二维的，使用 np.expand_dims 在其前面添加一个维度，以确保每个 EEG 试验都具有三维形状。
                # print("new_shape: ", eeg_trials_noisy[i].shape)
                
            if eeg_trials_noisy[i].shape[1] != n_elec:#检查每个 EEG 试验的通道数是否与 n_elec（电极数）相同
                print(f"problem because n_elec ({n_elec}) must be {eeg_trial_noisy.shape[1]}")
                eeg_trials_noisy[i] = np.swapaxes(eeg_trial_noisy, 1, 2)#如果通道数不匹配，使用 np.swapaxes 将 EEG 试验的通道与时间点之间的维度进行交换，以确保形状正确。
        
        if self.verbose:#检查是否应打印详细信息
            print(f'\nConvert EEG matrices to a single instance of mne.Epochs...')#指示正在将 EEG 数据矩阵转换为一个 mne.Epochs 实例
        ERP_samples_noisy = [np.mean(eeg_trial_noisy, axis=0) for eeg_trial_noisy in eeg_trials_noisy]#计算每个 EEG 试验的 ERP（事件相关电位），并将结果存储在 ERP_samples_noisy 列表中。ERP 是 EEG 信号的平均
        epochs = util.eeg_to_Epochs(ERP_samples_noisy, self.fwd_fixed, info=self.info)#将 ERP 数据转换为 mne.Epochs 对象，以便在后续的分析中使用。函数 util.eeg_to_Epochs 负责将数据转换为所需的格式
        return epochs
    
    def create_eeg_helper(self, eeg_sample, target_snr, beta):#将干净的脑电图（EEG）信号转换为带有噪声的 EEG 试验数据。该方法的输入是一个干净的 EEG 数据样本（`eeg_sample`）、目标信噪比（`target_snr`），以及控制噪声特性的 `beta` 参数
        #`create_eeg_helper` 方法的功能是将干净的 EEG 信号转换为带有噪声的 EEG 试验数据。噪声水平可以根据目标信噪比（`target_snr`）和 `beta` 参数进行控制。此方法还根据通道的线圈类型将噪声分配到不同的通道，以考虑可能具有不同尺度和特性的通道。最终，它返回带有噪声的 EEG 试验数据。
        ''' Helper function for EEG simulation that transforms a clean ,是一个辅助函数，用于 EEG 仿真
            M/EEG signal to a bunch of noisy trials.函数的主要功能，即将干净的脑电图（EEG）信号转换为带有噪声的多个试验数据。

        Parameters
        ----------
        eeg_sample : numpy.ndarray,是一个NumPy数组，包含干净的EEG信号数据。它具有两个维度，分别是电极数（electrodes）和时间点数（time_points）。
            data sample with dimension (electrodes, time_points)
        target_snr : float,是一个浮点数，表示目标的信噪比（Signal-to-Noise Ratio，信号与噪声的比值）。
            The target signal-to-noise ratio
        beta : float,是一个浮点数，表示 1/f**beta 噪声的 beta 指数。
            The beta exponent of the 1/f**beta noise

        '''
        
        assert len(eeg_sample.shape) == 2, 'Length of eeg_sample must be 2 (electrodes, time_points)'#这是一个断言语句，用于检查输入的 eeg_sample的形状是否为二维，即 (electrodes, time_points)。如果不是，将引发 AssertionError，指出 `eeg_sample` 的形状应该是二维的
        
        # Before: Add noise based on the GFP of all channels#在原来的实现中，噪声是基于所有通道的全局场电位（Global Field Power，GFP）来添加的，这意味着不同通道上的噪声幅度相同
        # noise_trial = self.add_noise(eeg_sample, target_snr, beta=beta)
        
        # NEW: ADD noise for different types of channels, separately#在新的实现中，噪声被分别添加到不同类型的通道上，因为不同类型的通道可能具有不同的信号幅度和噪声水平
        # since they can have entirely different scales.
        coil_types = [ch['coil_type'] for ch in self.info['chs']]#创建一个列表coil_types，其中包含了self.info['chs']中各个通道的线圈类型（coil_type）。
        coil_types_set = list(set(coil_types))#使用set将coil_types中的唯一线圈类型提取为一个列表 coil_types_set
        if len(coil_types_set)>1:#检查是否存在多个不同的线圈类型。如果是，将引发ValueError，指出模拟尝试使用多种通道类型，但应该只选择一个通道类型
            msg = f'Simulations attempted with more than one channel type \
                ({coil_types_set}) may result in unexpected behavior. Please \
                select one channel type in your data only'
            raise ValueError(msg)
            
        coil_types_set = np.array([int(i) for i in coil_types_set])#将 `coil_types_set` 列表中的元素转换为整数，并存储为 NumPy 数组。这是为了确保后续使用它们的索引操作不会出错
        
        coil_type_assignments = np.array(#创建一个 `coil_type_assignments` 数组，该数组为每个通道分配了一个线圈类型的索引。它使用 `np.where` 查找 `coil_types` 中的每个通道的线圈类型，并将其映射到 `coil_types_set` 数组中的索引
            [np.where(coil_types_set==coil_type)[0][0] 
                for coil_type in coil_types]
        )
        noise_trial = np.zeros(#创建一个与 `eeg_sample` 具有相同形状的全零数组 `noise_trial`，以存储带有噪声的 EEG 数据。
            eeg_sample.shape
        )

        for i, coil_type in enumerate(coil_types_set):#遍历唯一的线圈类型
            channel_indices = np.where(coil_type_assignments==i)[0]#获取具有特定线圈类型的通道的索引列表，将其存储在 `channel_indices` 中。
            #print(eeg_sample.shape)
            eeg_sample_temp = eeg_sample[channel_indices, :]#从原始eeg_sample中提取属于特定线圈类型的通道数据，存储在 `eeg_sample_temp` 中。
            noise_trial_subtype = self.add_noise(eeg_sample_temp, target_snr, beta=beta)#使用 `self.add_noise` 方法为特定线圈类型的通道数据 `eeg_sample_temp` 添加噪声，噪声水平由 `target_snr` 和 `beta` 参数确定。
            noise_trial[channel_indices, :] = noise_trial_subtype#将添加了噪声的通道数据 `noise_trial_subtype` 存储回原始 `noise_trial` 中，以构建最终的 EEG 试验数据
        #print("create_eeg_helper")
        return noise_trial
    
    def project_sources(self, sources):#这函数的目的是将源信号通过引导场（leadfield）投影为EEG数据
        ''' Project sources through the leadfield to obtain the EEG data.它的主要功能是将源信号通过引导场（leadfield）来生成EEG数据
        Parameters
        ----------
        sources : numpy.ndarray,要投影的源信号的Numpy数组。这是一个3D数组，具有形状 `(samples, dipoles, time points)`，表示多个样本、多个偶极子以及每个偶极子在不同时间点的信号
            3D array of shape (samples, dipoles, time points)
        
        Return
        ------

        '''
        n_samples = len(sources)#获取源信号的样本数，即第一维的长度。
        # n_elec, n_dipoles = self.leadfield.shape
        # eeg = np.zeros((n_samples, n_elec, n_timepoints))
        eeg = []#创建一个空列表 `eeg` 以存储投影后的EEG数据。
        # Swap axes to dipoles, samples, time_points
        # sources_tmp = np.swapaxes(sources, 0,1)
        # Collapse last two dims into one
        # short_shape = (sources_tmp.shape[0], 
            # sources_tmp.shape[1]*sources_tmp.shape[2])
        # sources_tmp = sources_tmp.reshape(short_shape)

        result = [np.matmul(self.leadfield, src.data) for src in sources]#遍历输入的源信号列表 `sources` 中的每个源，然后使用 `np.matmul` 函数将每个源信号与引导场 `self.leadfield` 相乘。这一行代码执行了投影操作，将源信号从3D数组 `(samples, dipoles, time points)` 投影到EEG数据。
        
        # Reshape result
        # result = result.reshape(result.shape[0], n_samples, n_timepoints)
        # swap axes to correct order
        # result = np.swapaxes(result,0,1)
        # Rescale
        # result /= scaler
        #print("project_sources")
        return result#返回投影后的EEG数据，这是一个列表，其中每个元素对应一个源信号的EEG数据。


    
    def add_noise(self, x, snr, beta=0):#为输入的信号 `x` 添加噪声，以达到指定的信噪比（SNR）
        #总之，这个方法的作用是将噪声添加到输入信号中，以达到指定的信噪比，同时保持信号和噪声的相对幅度。这是通过计算输入信号和噪声的全局场电平和均方根（RMS）值，然后根据计算结果缩放噪声信号以实现的。
        '''Add noise of given SNR to signal x.
        Parameters:参数 `x` 用于接收包含信号数据的二维NumPy数组，以便在后续的代码中向这些数据添加噪声。
        -----------
        x : numpy.ndarray, 2-dimensional numpy array of dims (channels, timepoints), 
        `channels` 表示数据中的信号通道数量, `timepoints` 表示每个信号通道上的时间点或时间采样数。
        Return:
        -------
        '''
    
        # This looks inconvenient but we need to make sure that there is no empty dimension for the powerlaw noise function.
        x_shape = (x.shape[0], np.clip(x.shape[1], a_min=2, a_max=np.inf).astype(int))#`x_shape` 是用于处理输入信号 `x` 形状的变量，确保 `x` 至少具有2个时间点，以便进行功率谱噪声计算。计算噪声的功率谱时，必须确保数据至少包含两个时间点，这样才能产生有意义的结果
        noise = cn.powerlaw_psd_gaussian(beta, x_shape)
        #print("old noise: ", noise.shape)
        noise = self.noise_generator.get_noise(n_time=x_shape[-1], exponent=beta)#noise_generator.get_noise()` 通过噪声生成器对象获取噪声信号。噪声的幅度受 `beta` 参数控制，它表示噪声的频谱特性，可以是0（白噪声）或1（粉噪声）。这部分生成的噪声与输入信号具有相同的形状。

        #print("new noise: ", noise.shape)
        #
        # In case we added another entry in the 2nd dimension we have to remove it here again.
        if x_shape[1] != x.shape[1]:
            noise=noise[:, :1]
    
        noise_gfp = np.std(noise, axis=0)# 计算噪声的全局场电平（GFP），表示噪声的全局振幅变化
        rms_noise = np.median(noise_gfp)  # rms(noise)，计算噪声信号的均方根（RMS）值。
        
        x_gfp = np.std(x, axis=0)# 计算输入信号 `x` 的全局场电平（GFP）。
        rms_x = np.median(x_gfp)  # np.mean(np.max(np.abs(x_gfp), axis=1))  # x.max()#计算输入信号 `x` 的均方根（RMS）值。
        if rms_x == 0:  
            # in case most of the signal is zero, e.g. when using biphasic pulses
            rms_x = abs(x_gfp).max()
        # rms_noise = rms(noise-np.mean(noise))
        noise_scaler = rms_x / (rms_noise*snr)#计算一个缩放因子，以使噪声信号的均方根（RMS）值乘以 `snr` 等于输入信号的均方根（RMS）值。这个缩放因子用于确保信噪比达到目标值。
        # print(f'rms_x = {rms_x}\nrms_noise = {rms_noise}\n\tScaling by {noise_scaler} to yield snr of {snr}')
        out = x + noise*noise_scaler  #计算最终的输出信号，将输入信号 `x` 与缩放后的噪声相加，以达到指定的信噪比（SNR）。
        #print("add_noise keshezhiSNR")
        return out

    def check_settings(self):#检查模拟的设置是否完整，如果有缺失的设置，则使用默认设置填充。
        ''' Check if settings are complete and insert missing 
            entries if there are any.
        '''
        if self.settings is None:#如果`self.settings`为`None`，则将其设置为`DEFAULT_SETTINGS`，`DEFAULT_SETTINGS`是一个包含默认模拟设置的字典
            self.settings = DEFAULT_SETTINGS
        self.fwd_fixed, self.leadfield, self.pos, _ = util.unpack_fwd(self.fwd)#解包前向模型(`self.fwd`)以获取所需的数据：`self.fwd_fixed`是一个包含固定导向的MNE前向模型对象，`self.leadfield`是前向模型的导联矩阵，`self.pos`是所有M/EEG导联的三维坐标，`_`是未使用的占位符。
        self.distance_matrix = cdist(self.pos, self.pos)#创建距离矩阵(`self.distance_matrix`)，它包含了每对电极之间的距离，使用`cdist`函数计算。

        # Check for wrong keys:检查设置中是否存在不在`DEFAULT_SETTINGS`中的键。如果存在不在`DEFAULT_SETTINGS`中的键，将引发`AttributeError`异常。
        for key in self.settings.keys():
            if not key in DEFAULT_SETTINGS.keys():
                msg = f'key {key} is not part of allowed settings. See DEFAULT_SETTINGS for reference: {DEFAULT_SETTINGS}'
                raise AttributeError(msg)
        
        # Check for missing keys and replace them from the DEFAULT_SETTINGS，遍历`DEFAULT_SETTINGS`的键，检查它们是否在设置中存在且不为`None`。如果某个键不存在或为`None`，则使用`DEFAULT_SETTINGS`中的默认值填充
        for key in DEFAULT_SETTINGS.keys():
            # Check if setting exists and is not None
            if not (key in self.settings.keys() and self.settings[key] is not None):
                self.settings[key] = DEFAULT_SETTINGS[key]
        
        if self.settings['duration_of_trial'] == 0:#检查`duration_of_trial`的值是否为0，如果是，将`self.temporal`设置为`False`，否则为`True`
            self.temporal = False
        else:
            self.temporal = True
        #print("Simulation check_settings")
        self.neighbors = self.calculate_neighbors()#调用`calculate_neighbors`方法来计算源之间的邻居关系，将结果存储在`self.neighbors`中。
        

    def calculate_neighbors(self):#用于计算源之间的邻居关系
        adj = mne.spatial_src_adjacency(self.fwd["src"], verbose=False).toarray().astype(int)#使用MNE工具函数`mne.spatial_src_adjacency`计算前向模型的源空间(`self.fwd["src"]`)的邻接矩阵(`adj`)，此矩阵表示源之间的邻接关系
        neighbors = np.array([np.where(a)[0] for a in adj], dtype=object)# 将邻接矩阵`adj`转换为NumPy数组，并创建一个包含每个源的邻居索引的数组(`neighbors`)。这个数组的每个元素是一个源的邻居索引数组，这些邻居共享相邻的源
        #print("Simulation calculate_neighbors")
        return neighbors

               
    @staticmethod
    def get_pulse(pulse_len):#get_pulse函数的目的是生成一个具有指定长度的脉冲信号。脉冲信号定义为一个正弦波的半个周期。pulse_len参数指定了脉冲信号的数据点数（长度）。
        ''' Returns a pulse of given length. 生成一个半个正弦波周期的脉冲信号，用于模拟一个简单的脉冲事件
        A pulse is defined as 
        half a revolution of a sine.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)#pulse_len参数是脉冲信号的数据点数（长度）。将pulse_len转换为整数，以确保长度是整数。
        freq = (1/pulse_len) / 2#计算信号的频率freq，使脉冲信号的周期为pulse_len，并将其缩短为半个周期
        time = np.arange(pulse_len)#使用numpy生成一个时间数组time，其中包含0到pulse_len-1的整数，用于表示脉冲信号的时间点

        signal = np.sin(2*np.pi*freq*time)#使用正弦函数生成脉冲信号，频率为freq
        #print("Simulation get_pulse")
        return signal#返回生成的信号
    
    @staticmethod
    def get_biphasic_pulse(pulse_len, center_fraction=1, temporal_jitter=0.):#生成一个具有指定长度的双相脉冲信号。
        ''' Returns a biphasic pulse of given length.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)#pulse_len参数指定了脉冲信号的数据点数（长度）。 将输入的脉冲长度转换为整数
        freq = (1/pulse_len) *center_fraction#/ 2# 计算频率，使脉冲信号在指定比例的时间内有非零值
        time = np.linspace(-pulse_len/2, pulse_len/2, pulse_len) # 生成时间点数组，从-pulse_len/2到pulse_len/2
        
        jitter = np.random.randn()*temporal_jitter#添加时间抖动
        signal = np.sin(2*np.pi*freq*time + jitter)#生成一个双相脉冲信号，包括中心部分和抖动
        crop_start = int(pulse_len/2 - pulse_len/center_fraction/2)#计算需要裁剪的起始位置
        crop_stop = int(pulse_len/2 + pulse_len/center_fraction/2)#计算需要裁剪的结束位置
        
        # signal[(time<-1) | (time>1)] = 0，将信号的中心部分以外的部分置零，以创建双相脉冲
        signal[:crop_start] = 0
        signal[crop_stop:] = 0
        signal *= np.random.choice([-1,1])#随机反转信号的极性
        #print("get_biphasic_pulse")
        return signal

    @staticmethod
    def get_from_range(val, dtype=int):#这个函数的主要目的是根据用户提供的输入值或范围，生成一个采样值，可以在指定范围内进行随机采样。如果`val`只是一个单一的值，那么函数就不会进行采样，而是直接返回这个值。同时，用户可以指定输出值的数据类型。
        ''' If list of two integers/floats is given this method outputs a value in between the two values.
        Otherwise, it returns the value.
        
        Parameters
        ----------
        val : list/tuple/int/float，val参数可以是单一的值，也可以是包含两个值的列表、元组或NumPy数组，表示一个范围。

        Return
        ------
        out : int/float

        '''
        # If input is a function -> call it and return the result
        if callable(val):# 如果输入是一个函数，调用它并返回结果
            return val()
   
        #函数使用random库中的randrange或uniform函数从指定的范围内（如果有范围）随机采样一个值，根据所选的数据类型设置随机数生成函数
        #dtype参数指定了输出值的数据类型，可以是int（整数）或float（浮点数）
        if dtype==int:
            rng = random.randrange
        elif dtype==float:
            rng = random.uniform
        else:
            msg = f'dtype must be int or float, got {type(dtype)} instead'
            raise AttributeError(msg)
        # 如果输入是列表、元组或NumPy数组
        if isinstance(val, (list, tuple, np.ndarray)):
            if val[0] == val[1]:
                out = dtype(val[0])
            else:
                out = rng(*val)
        else:
            # 如果输入本身只是一个单一的值，就返回它，不需要采样范围内的值
            # be returned in the desired dtype.
            out = dtype(val)
        #print("get_from_range")
        return out
    
    def save(self, file_name):#将模拟对象保存到磁盘上的文件中，以便稍后可以重新加载和使用。
        ''' Store the simulation object.
        Parameters
        ----------
        file_name : str
            Filename or full path to store the object to.用于指定要保存模拟对象的文件名或完整路径

        Example
        -------
         '''
        #sim = Simulation().simulate()#`Simulation()`创建了一个新的模拟对象，然后`.simulate()`方法执行了模拟过程，生成了模拟数据和结果。
        #sim.save('C/Users/User/Desktop/simulation.pkl')
        sim = Simulation().simulate()
        sim.save('F:/cw/simulation.pkl')
        
        with open(file_name, 'wb') as f:#打开一个文件以供写入。`file_name`是指定的文件名或路径，`'wb'`表示以二进制写入模式打开文件，这适用于保存二进制数据。
            pkl.dump(self, f)#使用Python的pickle模块将模拟对象`self`保存到打开的文件`f`中。pickle模块用于序列化Python对象，以便可以将它们保存到文件中或从文件中加载。在这里，`self`代表模拟对象本身，而`f`是用于保存对象的文件。
        #print("save")
    def to_nontemporal(self):#这个函数的主要目的是将内部数据表示从时序数据（temporal）转换为非时序数据（non-temporal）。
        ''' Converts the internal data representation from temporal to 
        non-temporal. 
        
        Specifically, this changes the shape of sources from a
        list of mne.sourceEstimate to a single mne.sourceEstimate in which the 
        time dimension holds a concatenation of timepoints and samples.

        The eeg data is reshaped from (samples, channels, time points) to  (samples*time points, channels, 1).
        重新构造 EEG 数据的形状，将其变为 (samples * time points, channels, 1)，即将时间点和样本维度连接起来，通道维度保持不变

        Parameters
        ----------
        

        Return
        ------
        self : esinet.Simulation
            Method returns itself for convenience

        '''
        if not self.temporal:
            print('This Simulation() instance is already non-temporal')
            return self

        self.temporal = False#设置 `temporal` 属性为 `False`，表示数据不再是时序数据。
        self.settings['duration_of_trial'] = 0#将 `duration_of_trial` 设置为 0，因为在非时序数据中不再有试验时序

        eeg_data_lstm = self.eeg_data.get_data()#从 `self.eeg_data` 中提取原始 EEG 数据（shape 为 (samples, channels, time points)）
        # Reshape EEG data执行以下操作来处理 EEG 数据
        eeg_data_single = np.expand_dims(np.vstack(np.swapaxes(eeg_data_lstm, 1,2)), axis=-1)
        # Pack into mne.EpochsArray object
        epochs_single = mne.EpochsArray(eeg_data_single, self.eeg_data.info, 
            tmin=self.eeg_data.tmin, verbose=False)#将重新构造的 EEG 数据封装成 `mne.EpochsArray` 对象，并使用原始 EEG 数据的信息和其他属性。
        # Store the newly shaped data
        self.eeg_data = epochs_single#存储新的 EEG 数据到 `self.eeg_data` 属性
        
        # Reshape Source data以下操作来处理源数据（source data）：重新构造源数据的形状，将其变为 (samples * time points, dipoles)，即将时间点和样本维度连接起来，二极子（dipoles）维度保持不变。
        source_data = np.vstack(np.swapaxes(np.stack(
            [source.data for source in self.source_data], axis=0), 1,2)).T
        # Pack into mne.SourceEstimate object，从 `self.source_data` 中提取源数据，这些源数据是 `mne.SourceEstimate` 对象的列表
        source_single = deepcopy(self.source_data[0])
        source_single.data = source_data
        self.source_data = source_single
        print("to_nontemporal")
        return self
        
    def shuffle(self):#对模拟数据中的样本进行随机重排（洗牌），以便在训练和测试模型时更好地评估性能。
        ''' Shuffle the simulated samples.通过这个方法，可以在进行机器学习任务时随机化数据，从而减少模型过拟合和提高性能评估的可靠性。'''
        sources = self.source_data
        epochs = self.eeg_data
        df = self.simulation_info#获取源数据、EEG 数据和模拟信息，并将它们分别存储在 `sources`、`epochs` 和 `df` 变量中。
        n_samples = len(epochs)#计算样本的数量。

        # Shuffle everything
        new_order = np.arange(n_samples).astype(int)#创建一个新的索引顺序new_order，该顺序是从 0 到 n_samples-1的整数数组，并对它们进行随机洗牌
        np.random.shuffle(new_order)#对新顺序进行洗牌操作，以确保样本的排列顺序是随机的。
        
        epochs = [epochs[i] for i in new_order]
        sources = [sources[i] for i in new_order]# 使用新的顺序重排 `epochs` 和 `sources` 列表中的样本，以便它们的顺序与 `new_order` 保持一致。这意味着现在样本的顺序已经被随机化。
        
        df = df.reindex(new_order)#更新模拟信息 `df`，以匹配新的样本顺序。

        # store back，将重排后的 `epochs`、`sources` 和 `df` 存回到模拟对象的相应属性中。
        self.eeg_data = epochs
        self.source_data = sources
        self.simulation_info = df
        print("shuffle")
        
    def crop(self, tmin=None, tmax=None, include_tmax=False, verbose=0):#主要用于裁剪 EEG 和源数据，以便只保留特定时间范围内的数据。
        #将源数据和 EEG 数据裁剪到指定的时间范围内，以便在进一步分析中仅关注特定时间段的数据
        eeg_data = []
        source_data = []# 创建两个空列表，用于存储裁剪后的 EEG 数据和源数据。
        if tmax is None:#检查是否提供了 `tmax` 参数，如果没有，则将 `tmax` 设置为 EEG 数据的默认 `tmax`。
            tmax = self.eeg_data[0].tmax
        for i in range(self.n_samples):#遍历模拟对象的每个样本（n_samples 是样本数）
            # print(self.eeg_data[i].tmax, tmax)
            cropped_source = self.source_data[i].crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)#对第 `i` 个样本的源数据进行裁剪，仅保留在指定时间范围内的数据
            cropped_eeg = self.eeg_data[i].crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax, verbose=verbose)#对第 `i` 个样本的 EEG 数据进行裁剪，同样只保留在指定时间范围内的数据。这里还可以选择是否输出详细的日志信息，根据 `verbose` 参数的设置。
            # min_crop = (1/cropped_source.sfreq)
            # while len(cropped_source.times) > len(cropped_eeg.times):

            #     # print(f"cropping: {len(cropped_source.times)}")
            #     tmax -= min_crop
            #     cropped_source = cropped_source.crop(tmin=tmin, tmax=tmax-min_crop)
            #     # print(f"cropped: {len(cropped_source.times)}")


            source_data.append( cropped_source )
            eeg_data.append( cropped_eeg )#将裁剪后的源数据和 EEG 数据添加到 `source_data` 和 `eeg_data` 列表中，分别对应每个样本。


        
        self.source_data = source_data
        self.eeg_data = eeg_data#处理完所有样本后，更新模拟对象的 `source_data` 和 `eeg_data`，以便包含已裁剪的数据。
        print("Simulation crop")
        return self


    def select(self, samples):#这个方法的目的是选择模拟对象中的样本子集。它接受一个参数 `samples`，该参数可以是整数、列表或元组。
        ''' Select subset of samples.
        Parameters
        ----------
        samples : int/list/tuple
            If type int select the given number of samples, if type list select indices given by list
        Return
        ------

        '''
        print("not implemented yet")
        return self
    
    def extents_to_orders(self, extents):#用于将源的直径（extents，以毫米为单位）转换为邻域级别（orders）的函数
        ''' Convert extents (source diameter in mm) to neighborhood orders.
        '''
        if self.diams is None:
            self.get_diams_per_order()#调用 `self.get_diams_per_order()` 方法来计算每个邻域级别的源直径，并将结果存储在 `self.diams` 中。
        if isinstance(extents, (int, float)):#如果extents是一个单一的整数或浮点数，它表示源的直径，函数将找到最接近的邻域级别，并返回该级别的索引。
            order = np.argmin(abs(self.diams-extents))
        else:#如果extents是一个包含两个整数或浮点数的列表或元组，它表示一组源的直径范围。函数将找到最接近的两个邻域级别，并返回这两个级别的索引组成的元组
            order = (np.argmin(abs(self.diams-extents[0])), np.argmin(abs(self.diams-extents[1])))
        #print("Simulation extents_to_orders")
        return order
    
    def get_diams_per_order(self):#用于计算每个邻域级别的估计源直径。函数不接受任何参数。计算每个邻域级别的估计源直径，并将结果存储在模拟对象的 `self.diams` 属性中，以供后续分析和模拟使用
        ''' Calculate the estimated source diameter per neighborhood order.
        '''
        diams = []#是一个空列表，用于存储每个邻域级别的估计源直径
        diam = 0#一个变量，初始化为0，用于存储当前估计的源直径
        order = 0#一个变量，初始化为0，表示邻域级别的顺序。
        while diam<60:#循环计算每个邻域级别的源直径，直到估计的源直径 `diam` 大于或等于100毫米为止
            #调用 `util.get_source_diam_from_order` 方法来计算给定级别的邻域的估计源直径，并将其添加到 `diams` 列表中。然后，`order` 值递增，以计算下一个邻域级别的源直径
            diam = util.get_source_diam_from_order(order, self.pos, dists=deepcopy(self.distance_matrix))
            diams.append( diam )
            order += 1
        self.diams = np.array(diams)#将 `diams` 列表转换为NumPy数组，并将其存储在 `self.diams` 中，以便以后的使用。
        #print("Simulation get_diams_per_order")
    
def get_n_order_indices(order, pick_idx, neighbors):#此函数用于通过选择邻域的邻居来执行区域生长，进行 `order` 次迭代
    #该函数用于执行区域生长操作，通过选择邻域的邻居来扩展邻域。它可以用于查找与初始种子点具有指定邻居级别的所有点的索引。
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.
    order：整数，表示要执行的迭代次数。如果 `order` 不是整数，将引发 AssertionError。
    `pick_idx`：整数，表示初始种子点的索引
    `neighbors`：一个列表，包含邻域的邻居索引。
    '''
    assert order == round(order), "Neighborhood order must be a whole number"
    order = int(order)#函数会检查 `order` 是否为整数，如果不是，会引发 AssertionError。
    if order == 0:
        return [pick_idx,]
    flatten = lambda t: [item for sublist in t for item in sublist]
    #print("y")
    current_indices = [pick_idx,]# 
    for cnt in range(order):# 函数使用一个 `for` 循环进行 `order` 次迭代，每次迭代都会选择邻域的邻居。在每次迭代中：函数首先通过索引列表 `current_indices` 找到当前邻域的索引。
        # current_indices = list(np.array( current_indices ).flatten())
        new_indices = [neighbors[i] for i in current_indices]
        new_indices = flatten( new_indices )# `new_indices` 列表中的索引表示当前邻域的邻居。
        current_indices.extend(new_indices)
        
        current_indices = list(set(current_indices))#函数将 `new_indices` 中的索引添加到 `current_indices` 中，并使用 `set` 来去除重复的索引。
    #print("Simulation get_n_order_indices")
    return current_indices# 函数在完成所有迭代后，返回包含 `current_indices` 中的索引的列表，这些索引表示在 `order` 次迭代后，与初始种子点具有某种邻居关系的所有点的索引。


class NoiseGenerator:#NoiseGenerator类
    ''' Generates multidimensional colored noise.
    NoiseGenerator 类用于生成具有指定频谱的有色噪声，然后将噪声传输到电极上，以便进行 EEG 模拟。这是 EEG 模拟中的重要步骤，以模拟噪声信号。
    Parameters
    ----------
    info : mne.Info
        The mne-python Info object, e.g. present in evoked.info
        NoiseGenerator类的构造函数，接受一个 mne.Info 对象作为参数。这个对象通常包含在 MNE-Python 中的 evoked.info 中。
    '''

    def __init__(self, info):#这是类 NoiseGenerator 的构造函数 __init__，它接受一个名为 info 的参数，该参数应该是一个 mne.Info 对象。在构造函数中，将传入的 info 存储为类属性 self.info，然后调用 self.prepare() 方法。
        '''
        Parameters
        ----------
        info : mne.Info
            The mne-python Info object, e.g. present in evoked.info
        '''
        self.info = info
        self.prepare()
        pass
    def prepare(self, resolution=16, k_neighbors=5):#prepare 方法用于准备一个规则的网格，该方法有两个可选参数 resolution 和 k_neighbors。
        ''' Prepare the regularly spaced grid.
        '''
        
        # n_time = sim.eeg_data[0].average().times.size
        # shape = (resolution, resolution, n_time)
        self.elec_pos = _find_topomap_coords(self.info, self.info.ch_names)#通过调用 _find_topomap_coords 函数，从 info 对象中提取出电极的位置信息，存储在 self.elec_pos 中
        x = np.linspace(self.elec_pos[:, 0].min(), self.elec_pos[:, 0].max(), num=resolution)
        y = np.linspace(self.elec_pos[:, 1].min(), self.elec_pos[:, 1].max(), num=resolution)#通过计算 x 和 y 的线性空间坐标，生成 resolution 个均匀分布的点。

        grid = np.stack(np.meshgrid(x,y, indexing='ij'), axis=0)
        grid_flat = grid.reshape(2, resolution**2)#创建一个网格，包括 x 和 y 的坐标信息，并将其展平为一个二维数组，存储在 grid 和 grid_flat 中。

        # grid_flat = grid.reshape(grid.shape[0], np.product(grid.shape[1:])).T
        neighbor_indices = np.stack([#neighbor_indices 存储了每个电极的最近邻索引。它通过迭代计算每个电极到 grid_flat 中所有点的欧氏距离，并将最近的 k_neighbors 个点的索引存储在数组中。
            np.argsort(np.sqrt(np.sum((grid_flat.T - coords)**2, axis=1)))[:k_neighbors] for coords in self.elec_pos
        ], axis=0)

        self.resolution = resolution
        self.grid = grid
        self.grid_flat = grid_flat
        self.k_neighbors = k_neighbors
        self.neighbor_indices = neighbor_indices#将准备好的数据存储为类属性，以备后续使用
        #print(" NoiseGenerator prepare")

    def get_noise(self, n_time, exponent=2):
        ''' Create colored noise of spectrum 1/f**exponent. The noise is first
        generated on an equally spaced grid and transferred the the electrodes
        using nearest-neighbor interpolation.该函数用于创建有色噪声。它首先生成具有指定频谱的噪声网格，然后使用最近邻插值将噪声传输到电极。

        '''

        noise_grid = util.create_n_dim_noise((self.resolution, self.resolution, n_time), exponent=exponent)#创建一个大小为 (resolution, resolution, n_time) 的多维噪声网格，该网格代表规则的噪声。
        noise_elec = np.zeros((self.elec_pos.shape[0], n_time))#初始化一个数组，用于存储噪声数据，数组的维度为 (电极数量, 时间点数量)
        for e, e_pos in enumerate(self.elec_pos):# 对每个电极进行遍历，将规则网格上的噪声插值到电极上
            for t in range(n_time):
                # neighbor_idc = np.argsort(np.sum((self.grid_flat.T - e_pos)**2, axis=1))[:self.k_neighbors]
                neighbor_idc = self.neighbor_indices[e,:]#获取最近邻的索引，这些最近邻用于进行插值。
                noise_transformed = np.mean(noise_grid[:, :, t].flatten()[neighbor_idc])#对于每个时间点 t，从规则噪声网格中提取邻居位置上的噪声，并取平均值。
                noise_elec[e, t] = noise_transformed#将插值后的噪声存储到噪声数组中的对应位置。
        #print("NoiseGenerator get_noise")       
        return noise_elec#返回生成的噪声数据，它代表了在电极位置上的有色噪声。