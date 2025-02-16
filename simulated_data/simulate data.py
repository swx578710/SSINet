import sys;
from esinet.evaluate import eval_mse
from esinet import Simulation,Net, util, evaluate
from esinet.util import calculate_source
from forward import create_forward_model, get_info, create_forward_model_gai, get_info2
from matplotlib import pyplot as plt
import numpy as np
import mne
from scipy.sparse.csgraph import laplacian
from tensorflow.keras.layers import Dense, TimeDistributed, Bidirectional, LSTM, multiply, Input, Dropout, Conv1D, Flatten, GRU
from esinet.evaluate import auc_metric

# Create generic Forward Model 设置频率
info = get_info(sfreq=256)
# info = get_info2(sfreq=256)
# 生成前向模型，ico2表示324个源，ico3表示1284个源，oct5表示2052个源,其间距为10mm ico4表示5124个源,
fwd = create_forward_model(info=info, sampling='ico3')
leadfield, pos = util.unpack_fwd(fwd)[1:3]
print(fwd)
print(leadfield.shape)
np.save('leadfield.npy', leadfield)
fwd.save('fwd.fif', overwrite=True)

# np.save('pos_128.npy', pos)
# 获取邻接矩阵和拉普拉斯算子
adjacency = mne.spatial_src_adjacency(fwd['src'], verbose=0).toarray()
laplace_operator = laplacian(adjacency)
# np.save('V.npy', adjacency)
# np.save('L.npy', laplace_operator)

settings = {
    'method': 'standard', # 模拟源信号时间的方式
    'number_of_sources': (1, 10), # 源的数量范围
    'extents':  (15, 35),  # 源的直径范围 单位是mm
    'amplitudes': (3, 10), # 源的单流大小范围，单位是nAm，前面的n表示纳，是10的-9次方，1 nAm = 0.001 µAm，默认(0.001, 100)
    'shapes': 'mixed', # 源的形状，可以是高斯分布或者均匀分布，以及二者混合分布
    'duration_of_trial': 0.1, #试次的时间长度，以秒为单位
    'sample_frequency': 256, # 数据的采样频率
    'target_snr': 1e99, # 目标信噪比范围，默认（2，20）
    'beta': (0.5, 1.5),
    'source_spread': "region_growing", # 源的空间分布
    'source_number_weighting': True, # 指示是否对源的数量进行加权，也就是从设定的范围内随机选择一个值作为源的数量
    'source_time_course': "random", # 源的时间过程类型是随机
}
sim = Simulation(fwd, info, settings=settings, verbose=True)
sim.simulate(n_samples=20000)

# 提取源信号
sources = sim.source_data
sources = [source.data for source in sources]
print(type(sources))
print(type(sources[0]))
y=np.array(sources)
print(type(y))
print(y.shape)

# 提取EEG信号
x=np.array(sim.eeg_data)
print(type(x))
print(x.shape)
x=x.squeeze()
print(x.shape)

# 保存x，y信号，用作神经网络训练
np.save('x10_clean.npy', x)
np.save('y10_clean.npy', y)

import pickle

# 指定要保存到的文件名
file_name = 'sim10_clean.pkl'

# 打开文件，使用二进制写模式
with open(file_name, 'wb') as f:
    # 使用pickle.dump()函数将对象保存到文件中
    pickle.dump(sim, f)