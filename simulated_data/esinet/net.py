import mne#导入MNE-Python库，这是一个用于处理脑电图（EEG）、磁力图（MEG）和其他神经物理学数据的Python库。
from mne.viz.topomap import (_setup_interp, _make_head_outlines, _check_sphere, 
    _check_extrapolate)#从MNE-Python库中导入用于制作头表地形图的相关功能。
from mne.channels.layout import _find_topomap_coords
import os#导入Python的标准os库，用于处理文件和目录路径等操作。
import tensorflow as tf#导入TensorFlow深度学习框架。
from tensorflow import keras#从TensorFlow库导入Keras，Keras是一个用于构建和训练神经网络的高级API。
from tensorflow.keras import layers#导入Keras中的不同类型的神经网络层。
from tensorflow.keras.layers import (LSTM, GRU, Dense, Flatten, Bidirectional, 
    TimeDistributed, InputLayer, Activation, Reshape, concatenate, Concatenate, 
    Dropout, Conv1D, Conv2D, multiply)
from tensorflow.keras import backend as K#导入Keras的后端（backend）模块，可用于进行底层操作。
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.sequence import pad_sequences#导入Keras的序列预处理功能，用于填充序列数据。
# from tensorflow.keras.utils import pad_sequences

from scipy.optimize import minimize_scalar#从SciPy库中导入用于标量最小化问题的最小化函数。
# import pickle as pkl
import dill as pkl#导入dill库并将其重命名为pkl，dill是Python的序列化库，用于保存和加载Python对象。
import datetime
# from sklearn import linear_model
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
from time import time
from tqdm import tqdm
from .util import util
#from . import util#导入当前目录下的自定义模块util、evaluate和losses.
from . import evaluate
from . import losses
from .custom_layers import BahdanauAttention, Attention#导入当前目录下的custom_layers模块，包括BahdanauAttention和Attention自定义层。

# Fix from: https://github.com/tensorflow/tensorflow/issues/35100#配置TensorFlow的GPU内存增长选项。在深度学习中，使用GPU来加速模型训练是常见的，但默认情况下，TensorFlow会占用GPU的所有内存。这可能会导致在同一台机器上运行多个TensorFlow会话时，GPU内存不足的问题。通过启用内存增长，TensorFlow会在需要时按需分配GPU内存，而不是一开始就占用整个GPU内存。这允许多个TensorFlow会话在同一台GPU上共享内存，并且只使用它们实际需要的内存
# devices = tf.config.experimental.list_physical_devices('GPU')#检查是否存在可用的物理GPU设备，通过调用来获取物理GPU列表
# if len(devices) > 0:#如果有可用的GPU设备
#     print(devices)
#     tf.config.experimental.set_memory_growth(devices, True)#则启用GPU内存增长选项,设置内存增长为True。
##TensorFlow会话将只在需要时分配和释放GPU内存，从而更有效地使用GPU资源。这对于同时运行多个TensorFlow会话的环境特别有用，以避免GPU内存浪费问题。
class Net:
    ''' The neural network class that creates and trains the model. 
    
    Attributes 类的属性
    ----------
    fwd : mne.Forward，这是一个类属性，表示`fwd`是一个`mne.Forward`类型的对象。`mne.Forward`通常是一个用于描述脑电源定位的数据结构。
        the mne.Forward forward model class.
    n_layers : int，这是一个类属性，表示`n_layers`是一个整数，用于指定神经网络中的隐藏层数。
        Number of hidden layers in the neural network.
    n_neurons : int，这是一个类属性，表示`n_neurons`是一个整数，用于指定每个隐藏层中的神经元数量。
        Number of neurons per hidden layer.
    activation_function : str，这是一个类属性，表示`activation_function`是一个字符串，用于指定每个完全连接层中使用的激活函数。
        The activation function used for each fully connected layer.
    n_jobs : int，这是一个类属性，表示`n_jobs`是一个整数，用于指定并行处理中使用的作业数/核心数。
        Number of jobs/ cores to use during parallel processing
    model : str，这是一个类属性，表示`model`是一个字符串，用于指定神经网络的体系结构。它可以有三个可能的取值：'auto'、'single'和'temporal'，分别表示自动选择模型、单个时间点模型和LSTM模型。
        Determines the neural network architecture.
            'auto' : automated selection for fully connected if training data 
                contains single time instances (non-temporal data)
            auto表示当训练数据包含单个时间实例（非时间序列数据）时，模型会自动选择完全连接神经网络体系结构，这意味着如果训练数据不涉及时间关系，模型将选择适用于非时间序列数据的神经网络结构。
            'single' : The single time instance model that does not learn 
                temporal relations.
            single表示使用单个时间实例的模型，该模型不会学习时间关系，这种模型适用于不考虑时间依赖性的情况，每个时间点的处理是相互独立的。
            'temporal' : The LSTM model which estimates multiples inverse 
                solutions in one go.
            temporal表示使用LSTM（Long Short-Term Memory）模型，该模型可以一次估计多个反演解决方案，这种模型适用于需要考虑时间依赖性的情况，例如处理时间序列数据，以捕获时间关系。

    Methods 类的方法
    -------
    fit : trains the neural network with the EEG and source data，fit方法用于使用脑电数据和源数据来训练神经网络模型，它执行训练过程。
    train : trains the neural network with the EEG and source data，与fit方法具有相同的功能，即使用脑电数据和源数据来训练神经网络模型。
    predict : perform prediciton on EEG data，predict方法用于对脑电数据执行预测操作，即使用已经训练好的模型来预测输出。
    evaluate : evaluate the performance of the model，evaluate方法用于评估模型的性能，通常使用一些性能指标来衡量模型的准确性。
    '''
    
    def __init__(self, fwd, n_dense_layers=1, n_lstm_layers=2, 
        n_dense_units=200, n_lstm_units=32, activation_function='tanh', 
        n_filters=64, kernel_size=(3,3), l1_reg=1e2, n_jobs=-1, model_type='auto', 
        scale_individually=True, rescale_sources='brent', 
        verbose=True):#这是构造函数的定义，它接受多个参数来初始化神经网络类的实例。注意model type有convdip、cnn、fc、lstm这些选择

        self._embed_fwd(fwd)#调用了`_embed_fwd`方法，用于将前向模型（fwd）嵌入到神经网络类的实例中。
        
        self.n_dense_layers = n_dense_layers#表示要在神经网络中包含的密集层的数量。
        self.n_lstm_layers = n_lstm_layers#表示要在神经网络中包含的LSTM层的数量。
        self.n_dense_units = n_dense_units#表示每个密集层中的神经元数量。
        self.n_lstm_units = n_lstm_units#表示每个LSTM层中的LSTM单元数量。
        self.l1_reg = l1_reg#将参数`l1_reg`的值赋给类属性`l1_reg`，表示L1正则化的强度。
        self.activation_function = activation_function#表示神经网络中使用的激活函数类型。
        self.n_filters = n_filters#表示卷积神经网络中的滤波器数量。
        self.kernel_size = kernel_size#表示卷积核的大小。
        # self.default_loss = tf.keras.losses.Huber(delta=delta)
        self.default_loss = 'mean_squared_error'  # losses.weighted_huber_loss，将默认的损失函数设置为均方误差损失。
        # self.parallel = parallel
        self.n_jobs = n_jobs#表示在并行处理中使用的作业数。
        self.model_type = model_type#表示神经网络的类型，可以是'auto'、'single'或'temporal'。
        self.compiled = False#表示模型尚未编译
        self.scale_individually = scale_individually#表示是否对输入数据进行单独缩放。
        self.rescale_sources = rescale_sources#表示源数据的重新缩放方法。可以是brent，也可以是rms
        self.verbose = verbose#表示是否显示详细信息。

    def _embed_fwd(self, fwd):
        ''' Saves crucial attributes from the Forward model.用于从给定的MNE Forward模型对象 (`fwd`) 中提取和保存关键属性。
        
        Parameters
        ----------
        fwd : mne.Forward
            The forward model object.一个MNE Forward模型对象，包含脑电源和传感器之间的前向模型信息。
        '''
        _, leadfield, _, _ = util.unpack_fwd(fwd)# 解包Forward模型，获取Forward模型的关键属性。
        self.fwd = deepcopy(fwd)
        self.leadfield = leadfield#获取 `leadfield`（引导场）的维度，包括通道数和偶极子数
        self.n_channels = leadfield.shape[0]
        self.n_dipoles = leadfield.shape[1]
        self.interp_channel_shape = (9,9)#设置插值通道形状：将插值通道的形状设置为 (9, 9)，并保存到 `self.interp_channel_shape` 属性中。
        print("Net _embed_fwd")
    
    @staticmethod
    def _handle_data_input(arguments):#用于处理输入数据。根据输入参数的类型和数量，提取 EEG 数据和源数据，以便后续的处理和分析。它接受一个名为 `arguments` 的元组作为参数。
        ''' Handles data input to the functions fit() and predict().用于处理输入数据，以供类中的 `fit` 和 `predict` 方法使用。
        
        Parameters
        ----------
        arguments : tuple，唯一参数，它是一个元组，包含输入给 `fit` 和 `predict` 方法的数据。
            The input arguments to fit and predict which contain data.
        
        Return
        ------
        eeg : mne.Epochs，这是一个 `mne.Epochs` 对象，表示 MEG/EEG 数据。
            The M/EEG data.
        sources : mne.SourceEstimates/list，这是一个 `mne.SourceEstimates` 对象或一个列表，表示源数据。
            The source data.

        '''
        #print(f"Type of arguments[0]: {type(arguments[0])}")
        print(f"Length of arguments: {len(arguments)}")

        if len(arguments) == 1:#检查是否只包含一个参数
            if isinstance(arguments[0], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
                eeg = arguments[0]
                sources = None#第一个参数是否属于 MNE-Python 中定义的多种数据类型，包括 `mne.Epochs`、`mne.Evoked`、`mne.io.Raw` 等。如果是其中的一种，说明只有 EEG 数据，而没有源数据，因此将 EEG 数据赋值给变量 `eeg`，并将源数据 `sources` 设置为 `None`。
                print("zhiyou EEG shuju")
                print("Type of input: EEG data")
            else:#如果第一个参数不属于上述任何一种类型，说明输入的参数可能是一个模拟对象（`simulation`）。在这种情况下，将从模拟对象中提取 EEG 数据和源数据，分别赋值给 `eeg` 和 `sources`
                simulation = arguments[0]
                eeg = simulation.eeg_data
                sources = simulation.source_data
                print("kennengshi moniduixiang simulation")
                print("Type of input: Simulation object")
                print("eeg:", eeg)
                print("sources:", sources)
                # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
                # raise AttributeError(msg)
                 
  
            
        elif len(arguments) == 2:#检查输入参数 `arguments` 的长度是否为2，即包含两个参数。如果是，说明第一个参数是 EEG 数据，第二个参数是源数据，分别赋值给 `eeg` 和 `sources`。
            eeg = arguments[0]
            sources = arguments[1]
            print("ganghaoshierwei fenpeigei eeg he sources")
            print("Type of input: EEG data and Source data")
        else:#既不是长度为1的情况，也不是长度为2的情况，则数据不符合要求
            msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
            print("shujubufuheyaoqiu")
            print("Invalid input")
            raise AttributeError(msg)
        print("Net _handle_data_input")
        return eeg, sources#返回处理后的 EEG 数据和源数据，以供类中的其他方法使用。

    def fit(self, *args, optimizer=None, learning_rate=0.001, 
        validation_split=0.1, epochs=60, metrics=None, device=None, 
        false_positive_penalty=2, delta=1., batch_size=8, loss=None, 
        sample_weight=None, return_history=False, dropout=0.2, patience=7, 
        tensorboard=False, validation_freq=1, revert_order=True):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        *args : 这是一个可变数量的参数，用于接受输入数据。这些参数可以是 EEG 数据（`mne.Epochs` 或 `numpy.ndarray`）和源数据（`mne.SourceEstimates` 或列表），或者它们可以是单个参数，即 `simulation` 对象。
            Can be either two objects: 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or only one:
                simulation : esinet.simulation.Simulation
                    The Simulation object

            - two objects，两个对象: EEG object (e.g. mne.Epochs) and Source object (e.g. mne.SourceEstimate)
        
        optimizer : tf.keras.optimizers，神经网络的优化器，用于反向传播。
            The optimizer that for backpropagation.
        learning_rate : float，学习率，控制训练中权重的更新速度。使用默认的 Adam 优化器，可以设置学习率。
            The learning rate for training the neural network
        validation_split : float，用于验证的数据比例。
            Proportion of data to keep as validation set.
        delta : int/float，Huber损失函数的 delta 参数。
            The delta parameter of the huber loss function
        epochs : int，训练的轮数，每个轮次都会使用一次所有的训练样本。
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str，监控训练性能的指标。如果未提供性能指标，使用默认的平均绝对误差（MAE）
            The metrics to be used for performance monitoring during training.
        device : str，设备名称，例如图形卡。
            The device to use, e.g. a graphics card.
        false_positive_penalty : float，定义假阳性预测的权重，以控制逆解的保守性。
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int，在反向传播期间同时计算误差的样本数。
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses，损失函数,使用默认的余弦相似度损失函数。
            The loss function.
        sample_weight : numpy.ndarray，可选的样本权重
            Optional numpy array of sample weights.
        return_history：是否返回训练历史信息。
        dropout：Dropout 层的丢弃率。
        patience：早停策略的耐心度。
        tensorboard：是否启用 TensorBoard 日志记录。
        validation_freq：验证频率。
        revert_order：是否在批量生成中反转样本顺序。表示在每个训练周期之前，对生成的样本进行一次反转。这样做的目的是为了引入更多的随机性，使模型能够更好地适应各种数据分布情况
        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''
        self.loss = loss
        self.dropout = dropout#Dropout 层的丢弃率。
    
        print("Net fit zhongde preprocess data")
        x_scaled, y_scaled = self.prep_data(args)#调用 `prep_data` 方法来准备输入数据 `x_scaled` 和目标数据 `y_scaled`。
        
        # Early stopping训练过程中使用 EarlyStopping 回调来实现早期停止策略，当验证损失不再减小时，提前停止训练。
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', verbose=self.verbose, patience=patience, restore_best_weights=True)#创建一个早期停止回调对象，以在训练期间监视验证损失，如果损失不再减小，则提前停止训练。
        if tensorboard:#如果 `tensorboard` 为真，创建 TensorBoard 回调对象，用于可视化训练进度。`log_dir` 是 TensorBoard 日志目录的路径。
            log_dir = "logs/fit/" + self.model.name + '_' + datetime.datetime.now().strftime("%m%d-%H%M")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            callbacks = [es, tensorboard_callback]
        else:
            callbacks = []#[es]
        if optimizer is None:#如果未提供优化器，使用默认的 Adam 优化器，可以设置学习率。
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            print("Net fit shiyong optimizer")
            # optimizer = tf.keras.optimizers.Adam(clipvalue=0.5)  # clipnorm=1.)
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                # momentum=0.35)
                
        if self.loss is None:#如果未提供损失函数，使用默认的余弦相似度损失函数。
            # self.loss = self.default_loss(weight=false_positive_penalty, delta=delta)
            # self.loss = 'mean_squared_error'
            self.loss = tf.keras.losses.CosineSimilarity()
            print("Net fit shiyong loss")


        elif type(loss) == list:#如果 `loss` 是一个列表，则使用列表中的函数来创建损失函数。
            self.loss = self.loss[0](*self.loss[1])
        if metrics is None:#如果未提供性能指标，使用默认的平均绝对误差（MAE）。
            # metrics = [self.default_loss(weight=false_positive_penalty, delta=delta)]
            metrics = ['mae']
            print("Net fit jisuan MAE zhibiao")
        
        # Compile if it wasnt compiled before
        if not self.compiled:#如果模型还没有编译，执行下面的步骤。
            self.model.compile(optimizer, self.loss, metrics=metrics)#编译神经网络模型，使用提供的优化器、损失函数和性能指标。
            self.compiled = True
            #print("shapes before fit: ", x_scaled.shape, y_scaled.shape)
        
        
        if self.model_type.lower() == 'convdip':#如果模型类型是 `'convdip'`，则执行以下步骤。
            print("interpolating for convdip...")
            elec_pos = _find_topomap_coords(self.info, self.info.ch_names)#获取电极位置信息。
            interpolator = self.make_interpolator(elec_pos, res=self.interp_channel_shape[0])#根据电极位置创建插值器，用于将 EEG 数据从传感器空间转换为源空间。
            x_scaled_interp = deepcopy(x_scaled)
            for i, sample in enumerate(x_scaled):#数据插值：对输入数据 `x_scaled` 中的每个样本进行插值操作，将其转换为源空间。这是通过将每个时间点的插值值连接成一个新的时间维度来完成的。
                list_of_time_slices = []
                for time_slice in sample:
                    time_slice_interp = interpolator.set_values(time_slice)()[::-1]
                    time_slice_interp = time_slice_interp[:, :, np.newaxis]
                    list_of_time_slices.append(time_slice_interp)
                x_scaled_interp[i] = np.stack(list_of_time_slices, axis=0)
                x_scaled_interp[i][np.isnan(x_scaled_interp[i])] = 0
            x_scaled = x_scaled_interp
            del x_scaled_interp
            print("yishanchu x_scaled_interp")
            print("\t...done")
            
        print(" Net fit model")
        n_samples = len(x_scaled)#获取样本数量
        stop_idx = int(round(n_samples * (1-validation_split)))#训练集停止索引
        gen = self.generate_batches(x_scaled[:stop_idx], y_scaled[:stop_idx], batch_size, revert_order=revert_order)#创建一个数据生成器，用于批量训练神经网络。
        steps_per_epoch = stop_idx // batch_size#确定每个训练周期的步数。
        validation_data = (pad_sequences(x_scaled[stop_idx:], dtype='float32'), pad_sequences(y_scaled[stop_idx:], dtype='float32'))#定义验证数据，用于在每个训练周期结束后评估模型的性能。


        
        if device is None:#如果没有指定设备，将使用 CPU 进行训练。
            # history = self.model.fit(x_scaled, y_scaled, 
            #     epochs=epochs, batch_size=batch_size, shuffle=True, 
            #     validation_split=validation_split, verbose=self.verbose, 
            #     callbacks=callbacks, sample_weight=sample_weight)
            #使用给定的生成器 `gen` 进行模型训练。`steps_per_epoch` 指定了每个训练周期的步数，`validation_data` 包含验证数据，`validation_freq` 指定了在多少个训练周期后进行验证。`sample_weight` 可以用于设置样本权重。训练过程的详细信息通过 `verbose` 控制。
            history = self.model.fit(x=gen, 
                    epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, verbose=self.verbose, callbacks=callbacks, 
                    sample_weight=sample_weight, validation_data=validation_data, 
                    validation_freq=validation_freq, workers=1)
        else:#如果指定了设备，将使用指定设备（例如，GPU）进行训练。
            with tf.device(device):
                # history = self.model.fit(x_scaled, y_scaled, 
                #     epochs=epochs, batch_size=batch_size, shuffle=True, 
                #     validation_split=validation_split, verbose=self.verbose,
                #     callbacks=callbacks, sample_weight=sample_weight)
                # history = self.model.fit_generator(gen)
                #类似于在 CPU 上的训练，但使用了指定的设备。
                history = self.model.fit(x=gen, 
                    epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, verbose=self.verbose, callbacks=callbacks, 
                    sample_weight=sample_weight, validation_data=validation_data, 
                    validation_freq=validation_freq, workers=1)
                

        #del x_scaled, y_scaled#在训练结束后，删除已使用的训练数据，以释放内存。
        #print(" Net fit yishanchu x_scaled, y_scaled")
        if return_history:
            return self, history#如果设置为返回训练历史，返回神经网络对象和训练历史。
        else:
            return self#如果不需要返回训练历史，只返回神经网络对象。
    @staticmethod
    def generate_batches(x, y, batch_size, revert_order=True):#主要用于在训练神经网络时进行数据批次的处理，创建一个用于训练神经网络的数据批次生成器，它确保数据按时间长度递增排列，并允许随机化数据批次的顺序，以增加训练的多样性
            # print('start generator')
            n_batches = int(len(x) / batch_size)#计算可以分成多少批次
            x = x[:int(n_batches*batch_size)]
            y = y[:int(n_batches*batch_size)]#确保输入数据 `x` 和目标数据 `y` 的长度是 `batch_size` 的整数倍，因为在批次处理中需要等长的数据。
            
            time_lengths = [x_let.shape[0] for x_let in x]#计算每个输入数据 `x_let` 的时间长度。
            idc = list(np.argsort(time_lengths).astype(int))#根据时间长度对数据进行排序，并返回排序后的索引列表。
            # print("len idc: ", len(idc), " idc: ", idc)
            
            x = [x[i] for i in idc]
            y = [y[i] for i in idc]#按照排序后的索引重新排列输入数据 `x` 和目标数据 `y`，以确保数据批次按时间长度递增排列。
            while True:#创建一个无限循环，用于不断生成数据批次。
                x_pad = []#用于存储填充后的输入数据。
                y_pad = []#用于存储填充后的目标数据。
                for batch in range(n_batches):#对每个批次进行迭代。
                    x_batch = x[batch*batch_size:(batch+1)*batch_size]#获取当前批次的输入数据。
                    y_batch = y[batch*batch_size:(batch+1)*batch_size]#获取当前批次的目标数据。
                    

                    if revert_order:#如果 `revert_order` 参数为真，随机反转数据批次的顺序
                        if np.random.randn()>0:
                            # x_batch = np.flip(x_batch, axis=1)
                            # y_batch = np.flip(y_batch, axis=1)
                            x_batch = [np.flip(xx, axis=1) for xx in x_batch]
                            y_batch = [np.flip(yy, axis=1) for yy in y_batch]
                    
                    
                    
                    x_padlet = pad_sequences(x_batch , dtype='float32' )
                    y_padlet = pad_sequences(y_batch , dtype='float32' )#使用pad_sequences函数对当前批次的输入和目标数据进行填充，以确保它们具有相同的长度。
                    
                        
                    x_pad.append( x_padlet )
                    y_pad.append( y_padlet )#将填充后的数据添加到 `x_pad` 和 `y_pad` 列表中。
                
                new_order = np.arange(len(x_pad))#创建一个索引列表，用于对数据批次的顺序进行随机化。
                np.random.shuffle(new_order)#随机打乱索引列表，以改变数据批次的顺序。
                x_pad = [x_pad[i] for i in new_order]
                y_pad = [y_pad[i] for i in new_order]#根据随机化后的索引列表重新排列数据批次。
                for x_padlet, y_padlet in zip(x_pad, y_pad):#对每个填充后的数据批次进行迭代，生成器的核心部分。
                    yield (x_padlet, y_padlet)#使用 `yield` 语句生成当前数据批次的输入和目标数据，从而创建一个生成器，可以在神经网络的训练过程中使用。
            print("Net generate_batches")

    def prep_data(self, args):
        ''' Train the neural network using training data (eeg) and labels (sources).使用输入的EEG数据（脑电信号）和标签数据（源估计或脑源活动）来训练神经网络模型。
        
        Parameters
        ----------
        *args : 
  可变数量的参数。可以传递以下两种形式之一的参数：
  1. `eeg`和`sources`：EEG数据和源数据，用于监督训练神经网络。
  2. `simulation`：一个`esinet.simulation.Simulation`对象，其中包含了EEG数据和源数据。
            Can be either two objects: 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or only one:
                simulation : esinet.simulation.Simulation
                    The Simulation object

            - two objects: EEG object (e.g. mne.Epochs) and Source object (e.g. mne.SourceEstimate)
        
        optimizer : tf.keras.optimizers，用于神经网络的优化器，即用于反向传播的优化算法。
            The optimizer that for backpropagation.
        learning_rate : float，训练神经网络时的学习率，控制参数更新的步长大小。
            The learning rate for training the neural network
        validation_split : float，数据集中用于验证的部分比例。通常，数据集会分为训练集和验证集，以便监控模型性能。
            Proportion of data to keep as validation set.
        delta : int/float，Huber损失函数的 delta 参数，用于定义损失函数的形状。
            The delta parameter of the huber loss function
        epochs : int，训练时迭代的次数，每个epoch包含对所有训练样本的一次遍历。
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str，用于监测模型性能的指标，可以是一个或多个字符串或指标函数
            The metrics to be used for performance monitoring during training.
        device : str，用于训练的设备，例如图形处理器（GPU）。
            The device to use, e.g. a graphics card.
        false_positive_penalty : float，定义假阳性预测的权重，用于调整反演解决方案的保守性或自由度。
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int，用于在反向传播期间同时计算误差的样本数量。
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses，用于训练的损失函数，通常是一个 TensorFlow/Keras 损失函数。
            The loss function.
        sample_weight : numpy.ndarray，可选的样本权重数组，用于为不同样本分配不同的重要性。
            Optional numpy array of sample weights.

        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''

        
        eeg, sources = self._handle_data_input(args)#从输入参数`args`中提取EEG数据和源数据，并存储在`eeg`和`sources`变量中。这些数据将被用于训练和测试神经网络。
        print("eeg:", eeg)
        print("sources:", sources)
        self.info = eeg[0].info#存储EEG数据的信息，通常包括通道信息、采样频率等。
        self.subject = sources.subject if type(sources) == mne.SourceEstimate \
            else sources[0].subject#确定数据的主题/受试者，通常是源数据的主题。

        # Ensure that the forward model has the same 
        # channels as the eeg object
        self._check_model(eeg)#用于确保EEG数据和模型具有相同的通道。如果不匹配，可能需要进行适当的调整或者发出警告。

        # Handle EEG input
        if (type(eeg) == list and isinstance(eeg[0], util.EPOCH_INSTANCES)) or isinstance(eeg, util.EPOCH_INSTANCES):#这个条件用于检查EEG数据的类型，以确定是否需要进行额外的处理。
            eeg = [eeg[i].get_data() for i, _ in enumerate(eeg)]#如果EEG数据的类型为`EPOCH_INSTANCES`，则从每个实例中提取EEG数据。
        else:#如果EEG数据不是`EPOCH_INSTANCES`类型，那么它被假定为`numpy.ndarray`，并且每个EEG示例中的第一个通道被选择。
            eeg = [sample_eeg[0] for sample_eeg in eeg]

        for i, eeg_sample in enumerate(eeg):#对于每个EEG样本，检查其数据形状（`eeg_sample.shape`）
            if len(eeg_sample.shape) == 1:#数据形状是一维的（例如 `(n_time,)`），则将其转换为二维数据，添加一个新的轴，以便数据具有形状 `(n_time, 1)`。这通常是因为神经网络模型期望输入数据是二维的，其中一个维度表示时间步长，另一个维度表示通道（或特征）。
                eeg[i] = eeg_sample[:, np.newaxis]
                print("eeg_sample.shape shi 1")
            if len(eeg_sample.shape) == 3:#三维，只选择其中的第一个时间步（例如 `eeg_sample[0]`），以便每个样本的数据形状变为 `(n_channels, n_time)`。
                eeg[i] = eeg_sample[0]
        #一般来说，神经网络通常期望输入数据是一个二维张量，其中一个轴表示时间步长，另一个轴表示通道或特征。在不同的应用中，EEG数据的形状可能会有所不同，因此这段代码旨在标准化数据的形状，以确保与神经网络模型的输入匹配。
        # check if temporal dimension has all-equal entries，这一行用于检查EEG数据的时间维度是否具有相同的长度，以便在训练中处理不同的时间长度。
        self.equal_temporal = np.all( np.array([sample_eeg.shape[-1] for sample_eeg in eeg]) == eeg[0].shape[-1])
        
        sources = [source.data for source in sources]#从源数据对象中提取源数据数组。

        # enforce shape: list of samples, samples of shape (channels/dipoles, time)，接下来的一系列assert语句用于确保数据的类型和维度与预期相匹配。
        assert len(sources[0].shape) == 2, "sources samples must be two-dimensional"
        assert len(eeg[0].shape) == 2, "eeg samples must be two-dimensional"
        assert type(sources) == list, "sources must be a list of samples"
        assert type(eeg) == list, "eeg must be a list of samples"
        assert type(sources[0]) == np.ndarray, "sources must be a list of numpy.ndarrays"
        assert type(eeg[0]) == np.ndarray, "eeg must be a list of numpy.ndarrays"
        

        # Scale sources
        y_scaled = self.scale_source(sources)#调用`scale_source`方法，用于缩放源数据。
        # Scale EEG
        x_scaled = self.scale_eeg(eeg)#调用`scale_eeg`方法，用于缩放EEG数据。

        # LSTM net expects dimensions to be: (samples, time, channels)，交换EEG和源数据的维度，以将它们调整为神经网络期望的形状。在这里，数据被交换为`(samples, time, channels)`的形状。
        x_scaled = [np.swapaxes(x,0,1) for x in x_scaled]
        y_scaled = [np.swapaxes(y,0,1) for y in y_scaled]
        
        # if self.model_type.lower() == 'convdip':
        #     x_scaled = [interp(x) for x in x_scaled]
        print("Net prep_data")
        return x_scaled, y_scaled#返回`x_scaled`和`y_scaled`，这是经过预处理的数据，准备用于神经网络的训练和测试。

    def scale_eeg(self, eeg):#对EEG数据进行缩放（标准化）的操作，
        ''' Scales the EEG prior to training/ predicting with the neural 
        network. 标准化的主要目的是确保不同特征之间的数值范围相似，从而帮助神经网络更有效地学习模型。在脑电数据中，不同的通道可能具有不同的幅度范围，通过标准化，可以使它们具有相似的均值和标准差。主要目的有三点：1.特征数值范围的一致性；2.更快的训练和收敛；3.降低模型对异常值的敏感性。

        Parameters，总结目的：通过标准化数据，确保每个通道在每个时间点上具有相似的数值范围，以帮助神经网络更好地处理脑电数据。这对于提高神经网络的训练效果和模型性能非常重要。
        ----------
        eeg : numpy.ndarray，一个3D矩阵，表示EEG数据，其形状为 `(samples, channels, time_points)`，samples：EEG数据样本的数量；channels：通道数；
 time_points：每个通道上的时间点数。
            A 3D matrix of the EEG data (samples, channels, time_points)
        
        Return
        ------
        eeg : numpy.ndarray
            Scaled EEG
        '''
        eeg_out = deepcopy(eeg)#输出的标准化后的EEG数据
        
        if self.scale_individually:#选择是否对每个样本进行独立的标准化或者进行总体标准化。
            for sample, eeg_sample in enumerate(eeg):#如果选择独立标准化，它首先应用常见平均参考（CAR）操作，减去每个时间点上所有通道的平均值以去除共同的噪声源。然后，对每个通道的每个时间点执行零均值化和单位方差化。
                # Common average ref:
                for time in range(eeg_sample.shape[-1]):
                    eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
                    # eeg_out[sample][:, time] /= np.max(np.abs(eeg_sample[:, time]))
                    eeg_out[sample][:, time] /= eeg_out[sample][:, time].std()
                    
                    
        else:#选择总体标准化，它会将所有样本的数据进行总体标准化，以确保整个数据集的振幅范围一致。
            for sample, eeg_sample in enumerate(eeg):
                eeg_out[sample] = self.robust_minmax_scaler(eeg_sample)
                # Common average ref:
                for time in range(eeg_sample.shape[-1]):
                    eeg_out[sample][:, time] -= np.mean(eeg_sample[:, time])
        print("Net scale_eeg")
        return eeg_out
    

    def scale_source(self, source):#对源信号进行缩放（标准化）的操作
        ''' Scales the sources prior to training the neural network.

        Parameters，标准化操作的目的是确保源信号的每个偶极子在每个时间点上具有相似的数值范围，从而帮助神经网络更好地处理源信号。
        ----------
        source : numpy.ndarray
            A 3D matrix of the source data (samples, dipoles, time_points)，一个3D矩阵，表示源数据，samples：源数据样本的数量；dipoles：源数；
 time_points：每个源上的时间点数。
        
        Return
        ------
        source : numpy.ndarray
            Scaled sources
        '''
        source_out = deepcopy(source)
        # for sample in range(source.shape[0]):
        #     for time in range(source.shape[2]):
        #         # source_out[sample, :, time] /= source_out[sample, :, time].std()
        #         source_out[sample, :, time] /= np.max(np.abs(source_out[sample, :, time]))
        for sample, _ in enumerate(source):
            # source_out[sample, :, time] /= source_out[sample, :, time].std()
            source_out[sample] /= np.max(np.abs(source_out[sample]))
        print("Net scale_source")
        return source_out
            
    @staticmethod
    def robust_minmax_scaler(eeg):#对源信号进行缩放的方法，它采用了一种鲁棒的最小-最大缩放方法。在这个方法中，源信号 `eeg` 首先计算了第25百分位数（lower）和第75百分位数（upper）。然后，对每个时间点的源信号进行缩放，以确保它们的值在0到1之间。缩放后的源信号保持了原始数据的分布特征，但将值范围缩放到一个固定范围内。
        lower, upper = [np.percentile(eeg, 25), np.percentile(eeg, 75)]
        print("Net robust_minmax_scaler")
        return (eeg-lower) / (upper-lower)

    def predict(self, *args, verbose=True):
        ''' Predict sources from EEG data.

        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data ,这样就是模拟的EEG数据
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data，这样就是表示与 EEG 数据相关联的源信号数据。
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object，它是一个 `esinet.simulation.Simulation` 对象，包含了 EEG 数据和源信号数据。
        
        Return
        ------
        outsource : either numpy.ndarray (if dtype='raw') or mne.SourceEstimate instance
        '''
        print(f"Type of args[0]: {type(args[0])}")
        print(f"Length of args: {len(args)}")
        eeg, _ = self._handle_data_input(args)#从传递给方法的参数 `*args` 中提取 EEG 数据，根据不同的输入数据类型，将 `eeg` 初始化为 `mne.Epochs` 或 `numpy.ndarray` 类型。这个方法还会返回一个未使用的参数，因此使用 `_` 来忽略它。

        if isinstance(eeg, util.EVOKED_INSTANCES):#通过一系列条件语句，根据输入数据的不同类型来处理 EEG 数据，确保数据格式一致。具体处理包括选择合适的数据通道，提取采样频率 (`sfreq`) 和开始时间 (`tmin`)，然后将数据转换为 numpy 数组。
            # Ensure there are no extra channels in our EEG
            eeg = eeg.pick_channels(self.fwd.ch_names)    

            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg.data
            # add empty trial dimension
            eeg = np.expand_dims(eeg, axis=0)
            if len(eeg.shape) == 2:
                # add empty time dimension
                eeg = np.expand_dims(eeg, axis=2)
        elif isinstance(eeg, util.EPOCH_INSTANCES):
            # Ensure there are no extra channels in our EEG
            eeg = eeg.pick_channels(self.fwd.ch_names)
            eeg.load_data()

            sfreq = eeg.info['sfreq']
            tmin = eeg.tmin
            eeg = eeg._data
        elif isinstance(eeg, list) and isinstance(eeg[0], util.EPOCH_INSTANCES):
            sfreq = eeg[0].info['sfreq']
            tmin = eeg[0].tmin
            eeg = [e.get_data()[0] for e in eeg]
            
        # else:
        #     msg = f'eeg must be of type <mne.EvokedArray> or <mne.epochs.EpochsArray>; got {type(eeg)} instead.'
        #     raise ValueError(msg)
        # Prepare EEG to ensure common average reference and appropriate scaling
        # eeg_prep =  self._prep_eeg(eeg)
        eeg_prep = self.scale_eeg(deepcopy(eeg))#调用 `scale_eeg` 方法，对 EEG 数据进行缩放操作。这个操作包括对每个样本的每个时间点的 EEG 信号进行标准化处理。
        
        # Reshape to (samples, time, channels)
        eeg_prep = [np.swapaxes(e, 0, 1) for e in eeg_prep]#将 EEG 数据重新排列，以使其形状变为 `(samples, time, channels)`。
        if self.model_type.lower() == 'convdip':#如果神经网络模型类型是 'convdip'，则进行额外的插值处理。它会使用 `interpolator` 对数据进行插值，并处理可能的 NaN 值。最终，`eeg_prep` 包含了用于源信号预测的 EEG 数据
            print("interpolating for convdip...")
            elec_pos = _find_topomap_coords(self.info, self.info.ch_names)
            interpolator = self.make_interpolator(elec_pos, res=self.interp_channel_shape[0])
            eeg_prep_interp = deepcopy(eeg_prep)
            for i, sample in tqdm(enumerate(eeg_prep)):
                list_of_time_slices = []
                for time_slice in sample:
                    time_slice_interp = interpolator.set_values(time_slice)()[::-1]
                    time_slice_interp = time_slice_interp[:, :, np.newaxis]
                    list_of_time_slices.append(time_slice_interp)
                eeg_prep_interp[i] = np.stack(list_of_time_slices, axis=0)
                eeg_prep_interp[i][np.isnan(eeg_prep_interp[i])] = 0
            eeg_prep = eeg_prep_interp
            del eeg_prep_interp
            # print("shape of eeg_prep before prediciton: ", eeg_prep[0].shape)
            predicted_sources = self.predict_sources_interp(eeg_prep)
        else:
            # Predicted sources all in one go
            # print("shape of eeg_prep before prediciton: ", eeg_prep[0].shape)
            predicted_sources = self.predict_sources(eeg_prep)       

        # Rescale Predicitons根据用户选择的 `rescale_sources` 方法（'brent' 或 'rms'），对预测的源信号进行重新缩放。不同的方法可能采用不同的缩放策略。
        if self.rescale_sources.lower() == 'brent':
            predicted_sources_scaled = self._solve_p_wrap(predicted_sources, eeg)
        elif self.rescale_sources.lower() == 'rms':
            predicted_sources_scaled = self._scale_p_wrap(predicted_sources, eeg)
        else:
            print("Warning: <rescale_sources> is set to {self.rescale_sources}, but needs to be brent or rms. Setting to default (brent)")
            predicted_sources_scaled = self._solve_p_wrap(predicted_sources, eeg)



        # Convert sources (numpy.ndarrays) to mne.SourceEstimates objects
        if verbose>0:#设置了 `verbose` 大于 0，打印模拟 EEG 数据和估计的 EEG 数据的形状，并计算每个样本的残差方差（`residual_variances`），用于评估估计结果的拟合程度。
            eeg_hat = list()
            for predicted_source in predicted_sources_scaled:
                eeg_hat.append( self.leadfield @ predicted_source )
            print("True eeg shape: ", np.stack(eeg, axis=0).shape)
            print("est eeg shape: ", np.stack(eeg_hat, axis=0).shape)
            
            residual_variances = [round(self.calc_residual_variance(M_hat, M), 2) for M_hat, M in zip(eeg_hat, eeg)]
            print(f"Residual Variance(s): {residual_variances} [%]")

        predicted_source_estimate = [
            util.source_to_sourceEstimate(predicted_source_scaled, self.fwd, \
                sfreq=sfreq, tmin=tmin, subject=self.subject) \
                for predicted_source_scaled in predicted_sources_scaled]
        #print(dir(predicted_sources_scaled))
        print("Net predict")
        return predicted_source_estimate#预测的源信号数据，每个元素都是 `mne.SourceEstimate` 对象，以满足不同分析需求。

    def calc_residual_variance(self, M_hat, M):#计算预测源信号和真实源信号之间的残差方差。这是通过将差的平方和与真实源信号的平方和的比值乘以100来表示的，用于量化拟合的质量。
        print("Net calc_residual_variance")
        return 100 * np.sum( (M-M_hat)**2 ) / np.sum(M**2)
        

    def predict_sources(self, eeg):#用于从 EEG 数据预测源信号。它接受 EEG 数据，该数据的形状是 `(samples, channels, time)`。
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg[0].shape)==2, 'eeg must be a list of 2D numpy array of dim (channels, time)'#检查每个 EEG 数据样本的形状是否为二维，以确保处理正确的数据类型

        predicted_sources = [self.model.predict(e[np.newaxis, :, :], verbose=self.verbose)[0] for e in eeg]#循环遍历每个 EEG 数据样本，并使用神经网络模型进行预测，用model进行预测，通过predict方法完成
            
        # predicted_sources = np.swapaxes(predicted_sources,1,2)
        predicted_sources = [np.swapaxes(src, 0, 1) for src in predicted_sources]#存储预测结果，每个元素是一个源信号的预测
        print("shape of predicted sources: ", predicted_sources[0].shape)
        print("Net predict_sources ")
        return predicted_sources

    def predict_sources_interp(self, eeg):
        ''' Predict sources of 3D EEG (samples, channels, time) by reshaping 
        to speed up the process.
        用于神经网络模型类型为 'convdip' 时的额外插值操作。它接受形状为 `(samples, time, height, width, 1)` 的 EEG 数据。同样，方法检查数据的形状是否正确，然后循环遍历每个 EEG 数据样本，并使用神经网络模型进行预测。最终，将预测的源信号整理为与输入数据相同的形状。
        Parameters
        ----------
        eeg : numpy.ndarray
            3D numpy array of EEG data (samples, channels, time)
        '''
        assert len(eeg[0].shape)==4, 'eeg must be a list of 4D numpy array of dim (time, height, width, 1)'#检查EEG数据是否是四维

        predicted_sources = [self.model.predict(e[np.newaxis, :, :], verbose=self.verbose)[0] for e in eeg]
            
        # predicted_sources = np.swapaxes(predicted_sources,1,2)
        predicted_sources = [np.swapaxes(src, 0, 1) for src in predicted_sources]
        print("shape of predicted sources: ", predicted_sources[0].shape)
        print("Net predict_sources_interp ")
        return predicted_sources

    def _scale_p_wrap(self, y_est, x_true):#接受两个参数 `y_est` 和 `x_true`，分别表示预测的源信号和真实 EEG 数据。
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''

        # assert len(y_est[0].shape) == 3, 'Sources must be 3-Dimensional'
        # assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'
        y_est_scaled = deepcopy(y_est)

        for trial, _ in enumerate(x_true):
            for time in range(x_true[trial].shape[-1]):
                scaled = self.scale_p(y_est[trial][:, time], x_true[trial][:, time])#调用 `scale_p` 方法，它用于缩放预测的源信号，以使其与真实 EEG 数据匹配。前面是预测的源信号的某个时间点，后面是真实 EEG 数据的相应时间点。
                y_est_scaled[trial][:, time] = scaled#将缩放后的源信号存储回 `y_est_scaled`
        print("Net _scale_p_wrap")
        return y_est_scaled

    def _solve_p_wrap(self, y_est, x_true):#它的操作与 `_scale_p_wrap` 类似，不同之处在于它调用的是 `solve_p` 方法，用于执行不同的缩放操作。
        ''' Wrapper for parallel (or, alternatively, serial) scaling of 
        predicted sources.
        '''
        # assert len(y_est.shape) == 3, 'Sources must be 3-Dimensional'
        # assert len(x_true.shape) == 3, 'EEG must be 3-Dimensional'

        y_est_scaled = deepcopy(y_est)

        for trial, _ in enumerate(x_true):
            for time in range(x_true[trial].shape[-1]):
                scaled = self.solve_p(y_est[trial][:, time], x_true[trial][:, time])
                y_est_scaled[trial][:, time] = scaled
        print("Net _solve_p_wrap")
        return y_est_scaled

    # @staticmethod
    # def _prep_eeg(eeg):
    #     ''' Takes a 3D EEG array and re-references to common average and scales 
    #     individual scalp maps to max(abs(scalp_map) == 1
    #     '''
    #     assert len(eeg.shape) == 3, 'Input array <eeg> has wrong shape.'

    #     eeg_prep = deepcopy(eeg)
    #     for trial in range(eeg_prep.shape[0]):
    #         for time in range(eeg_prep.shape[2]):
    #             # Common average reference
    #             eeg_prep[trial, :, time] -= np.mean(eeg_prep[trial, :, time])
    #             # Scaling
    #             eeg_prep[trial, :, time] /= np.max(np.abs(eeg_prep[trial, :, time]))
    #     return eeg_prep

    def evaluate_mse(self, *args):#评估模型的均方误差（Mean Squared Error，MSE）。均方误差是一种常用的评估指标，用于度量模型的预测与真实数据之间的差异。
        ''' Evaluate the model regarding mean squared error，
        根据模型的预测结果和真实数据计算均方误差，以评估模型的性能。均方误差是一个常见的回归问题评估指标，它可以帮助确定模型的预测能力和准确性。
        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object

        Return，返回每个样本的均方误差
        ------
        mean_squared_errors : numpy.ndarray
            The mean squared error of each sample

        Example，这个例子说明了怎么使用这个方法来计算均方误差并打印其均值
        -------
        net = Net()
        net.fit(simulation)
        mean_squared_errors = net.evaluate_mse(simulation)
        print(mean_squared_errors.mean())
        '''
 
        
                
        print(f"Type of args[0]: {type(args[0])}")
        print(f"Length of args: {len(args)}")        
        simulation = args[0]
        eeg = simulation.eeg_data
        sources = simulation.source_data
        print("kennengshi moniduixiang simulation")
        print("eeg:", eeg)
        print("sources:", sources)
        #eeg, sources = self._handle_data_input(args)#用于处理传入的参数 `args`，并将结果分配给 `eeg` 和 `sources` 变量
        #eeg, _ = self._handle_data_input(args)
        y_hat = self.predict(eeg)#调用模型的 `predict` 方法，使用输入的 EEG 数据 `eeg` 来获取预测的源信号数据。`y_hat` 将包含模型的预测源数据。
        #对 `y_hat` 和 `y` 进行数据格式的处理，确保它们具有相同的维度，以便进行均方误差计算。
        '''
        if type(y_hat) == list:
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        else:
            y_hat = y_hat.data

        if type(sources) == list:
            y = np.stack([y.data for y in sources], axis=0)
        else:
            y = sources.data
        '''
        if isinstance(y_hat, list):
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        elif isinstance(y_hat, mne.SourceEstimate):
            y_hat = y_hat.data

        if isinstance(sources, list):
            y = np.stack([y.data for y in sources], axis=0)
        elif isinstance(sources, mne.SourceEstimate):
            y = sources.data

        if len(y_hat.shape) == 2:
            y = np.expand_dims(y, axis=0)
            y_hat = np.expand_dims(y_hat, axis=0)

        mean_squared_errors = np.mean((y_hat - y)**2, axis=1)#计算每个样本中 `y_hat` 与 `y` 的差异，然后对这些差异的平方取平均。最后，它在每个样本上计算 MSE，并将结果存储在 `mean_squared_errors` 数组中。
        print("Net evaluate_mse")
        return mean_squared_errors#返回计算得到的均方误差数组 


    def evaluate_nmse(self, *args):
        ''' Evaluate the model regarding normalized mean squared error，这个函数用于评估模型性能，特别是通过计算标准化均方误差来度量预测源信号与实际源信号之间的拟合质量。
        
        Parameters
        ----------
        *args : 
            Can be either 
                eeg : mne.Epochs/ numpy.ndarray
                    The simulated EEG data，可以是 MNE 库中的 `mne.Epochs` 对象或 NumPy 数组，代表模拟的 EEG 数据。
                sources : mne.SourceEstimates/ list of mne.SourceEstimates
                    The simulated EEG data，可以是 MNE 库中的 `mne.SourceEstimates` 对象或 `list`，代表模拟的 EEG 数据的源信号
                or
                simulation : esinet.simulation.Simulation
                    The Simulation object，可以是 `esinet.simulation.Simulation` 对象，代表整个模拟。

        Return
        ------
        normalized_mean_squared_errors : numpy.ndarray
            The normalized mean squared error of each sample

        Example，如何输出标准化后的均方误差
        -------
        net = Net()
        net.fit(simulation)
        normalized_mean_squared_errors = net.evaluate_nmse(simulation)#一个 NumPy 数组，包含每个样本的标准化均方误差。
        print(normalized_mean_squared_errors.mean())
        '''

        eeg, sources = self._handle_data_input(args)#通过调用 `_handle_data_input` 方法，从传递给函数的参数中提取 EEG 数据和源信号数据。
        
        y_hat = self.predict(eeg)#使用模型的 `predict` 方法对 EEG 数据进行预测，得到预测的源信号 `y_hat`。
        #y_hat = net.predict(sim)
        if type(y_hat) == list:#确保源信号数据 `y_hat` 和实际源信号数据 `y` 的格式一致，以便进行计算。
            y_hat = np.stack([y.data for y in y_hat], axis=0)
        else:
            y_hat = y_hat.data

        if type(sources) == list:
            y = np.stack([y.data for y in sources], axis=0)
        else:
            y = sources.data
        
        if len(y_hat.shape) == 2:
            y = np.expand_dims(y, axis=0)
            y_hat = np.expand_dims(y_hat, axis=0)

        for s in range(y_hat.shape[0]):#循环遍历 `y_hat` 和 `y`，对每个样本的每个时间点进行标准化，以确保它们的幅度范围在[-1, 1]之间。
            for t in range(y_hat.shape[2]):
                y_hat[s, :, t] /= np.max(np.abs(y_hat[s, :, t]))
                y[s, :, t] /= np.max(np.abs(y[s, :, t]))
        
        normalized_mean_squared_errors = np.mean((y_hat - y)**2, axis=1)#计算标准化均方误差，将结果存储在 `normalized_mean_squared_errors` 
        print("Net evaluate_nmse")
        return normalized_mean_squared_errors#返回以上计算的数组

    def _build_model(self):#用于构建神经网络模型的架构
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API. Depending on the input data this 
        function will either build:使用 TensorFlow 的 Keras 库构建神经网络架构，具体取决于输入数据的类型。

        (1) A simple single hidden layer fully connected ANN for single time instance data，单个时间点数据，构建单隐藏层全连接的ANN网络
        (2) A LSTM network for spatio-temporal prediction，涉及时空关系的输入，构建LSTM用于时空预测
        '''
        if self.model_type.lower() == 'convdip':
            self._build_convdip_model()#用于构建 ConvDip 模型。
        elif self.model_type.lower() == "cnn":
            self._build_cnn_model()#用于构建 CNN 模型。
        elif self.model_type.lower() == 'fc':
            self._build_fc_model()#用于构建全连接（Fully Connected）模型。
        elif self.model_type.lower() == 'lstm':
            self._build_temporal_model()#用于构建 LSTM 模型。
        else:
            self._build_temporal_model()#如果没有匹配的模型类型，也调用 `_build_temporal_model` 方法。也就是LSTM。

        if self.verbose:
            self.model.summary()#如果 `verbose` 参数为真（非零），则执行self.model.summary()`，打印模型的摘要，包括模型的架构、层次结构和参数数量等信息。
    
    
    def _build_temporal_model(self):#构建LSTM网络模型，用于时序数据建模
        ''' Build the temporal artificial neural network model using LSTM layers.
        '''
        name = "LSTM Model"
        self.model = keras.Sequential(name=name)#用于顺序堆叠神经网络层
        tf.keras.backend.set_image_data_format('channels_last')#设置图像数据的通道顺序为'channels_last'，这对于时序数据是合适的。
        input_shape = (None, self.n_channels)#定义输入数据的形状，其中None表示时间步数可以是任意的
        
        # LSTM layers
        if isinstance(self.n_lstm_units, (tuple, list)):#如果n_lstm_units是元组或列表，将其转换为单个整数值，以确定LSTM层的单元数。
            self.n_lstm_units = self.n_lstm_units[0]
        # Dropout
        if isinstance(self.dropout, (tuple, list)):#如果dropout是元组或列表，将其转换为单个浮点值，以确定应用于Dropout层的丢弃率。
            self.dropout = self.dropout[0]

        # Model Architecture
        inputs = tf.keras.Input(shape=input_shape, name='Input')#创建一个输入层，接受形状为input_shape的输入数据。
        ## FC-Path
        fc1 = TimeDistributed(Dense(self.n_dense_units, 
                    activation=self.activation_function), 
                    name='FC1')(inputs)#创建一个时间分布的全连接层（TimeDistributed），接受输入数据，并应用具有self.n_dense_units单元和激活函数self.activation_function的全连接操作。将其命名为'FC1'。
        fc1 = Dropout(self.dropout)(fc1)#应用Dropout层以防止过拟合。
        direct_out = TimeDistributed(Dense(self.n_dipoles, 
            activation="linear"),
            name='FC2')(fc1)#创建一个时间分布的全连接层，用于生成直接输出。该层包含self.n_dipoles个单元，激活函数为线性激活。将其命名为'FC2'。
        # LSTM Path
        lstm1 = Bidirectional(LSTM(self.n_lstm_units, return_sequences=True, 
            input_shape=(None, self.n_dense_units), dropout=self.dropout), 
            name='LSTM1')(fc1)#创建一个双向LSTM层，该层接受fc1的输出，有self.n_lstm_units个单元，返回完整的时序数据（return_sequences=True），并应用Dropout。将其命名为'LSTM1'。
        mask = TimeDistributed(Dense(self.n_dipoles, 
                    activation="sigmoid"), 
                    name='Mask')(lstm1)#创建一个时间分布的全连接层，用于生成遮罩（mask）以在直接输出上应用。该层包含self.n_dipoles个单元，激活函数为sigmoid。将其命名为'Mask'。
        
        # Combination
        multi = multiply([direct_out, mask], name="multiply")#将直接输出和遮罩相乘，得到最终的多路输出。这是一个元素级的相乘操作。
        self.model = tf.keras.Model(inputs=inputs, outputs=multi, name='Contextualizer')# 这一行代码创建了一个Keras模型对象 `self.model`。`inputs` 是模型的输入张量，outputs` 是模型的输出张量，这里是前一步创建的 `multi`，name='Contextualizer'` 用于给模型指定一个名称。

        if self.l1_reg is not None:#检查 `self.l1_reg` 是否为非空，即是否有L1正则化参数传递给这个对象。
            self.model.add_loss(self.l1_reg * self.l1_sparsity(multi))#将L1正则化损失项添加到模型中。L1正则化通常用于鼓励模型中的权重参数变得稀疏（接近于零）。`self.l1_reg` 是一个L1正则化的超参数，它通常用于控制正则化的强度。
        
    def _build_fc_model(self):
        ''' Build the temporal artificial neural network model using LSTM layers.构建一个具有LSTM层和前馈神经网络层的神经网络模型，用于处理时间序列数据。模型包括输入层、LSTM层、前馈神经网络层、输出层和L1正则化损失。
        '''
        # self.model = keras.Sequential(name=name)
        tf.keras.backend.set_image_data_format('channels_last')
        input_shape = (None, self.n_channels)#一个时间序列数据的形状，其中第一个维度是时间步数（可以是任何值，用`None`表示），第二个维度是通道数
        # self.model.add(InputLayer(input_shape=input_shape, name='Input'))
        inputs = tf.keras.Input(shape=input_shape, name='Input_FC')#创建一个Keras输入层，定义了输入数据的形状和名称。
        
  
        if not isinstance(self.dropout, (tuple, list)):#检查 `self.dropout` 是否是一个元组或列表，如果不是，则创建一个长度为 `self.n_lstm_layers` 的列表 `dropout`，并将 `self.dropout` 的值复制到每个元素中。
            dropout = [self.dropout]*self.n_lstm_layers
        else:
            dropout = self.dropout
        
    
        # Hidden Dense layer(s):构建了一个或多个时间分布式稠密层（TimeDistributed Dense Layer），这些层用于对输入数据进行前馈神经网络处理。
        if not isinstance(self.n_dense_units, (tuple, list)):
            self.n_dense_units = [self.n_dense_units] * self.n_dense_layers
        
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*self.n_dense_layers
        else:
            dropout = self.dropout
        
        add_to = inputs
        for i in range(self.n_dense_layers):#用于构建多个稠密层，每个层都具有相同的单元数和激活函数。
            dense = TimeDistributed(Dense(self.n_dense_units[i], 
                activation=self.activation_function), name=f'FC_{i}')(add_to)#`dense` 是当前稠密层的输出，它将作为下一层的输入。
            dense = Dropout(dropout[i], name=f'Drop_{i}')(dense)# `Dropout` 层用于应用丢弃正则化
            add_to = dense

        # Final For-each layer:
        out = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='FC_Out')(dense)#构建了一个最终的时间分布式稠密层，输出的形状是 `(None, self.n_dipoles)`，激活函数是线性的。
        self.model = tf.keras.Model(inputs=inputs, outputs=out, name='FC_Model')#创建了一个Keras模型对象 `self.model`，将输入层 `inputs` 和最终输出层 `out` 组合在一起，给模型一个名称。

        if self.l1_reg is not None:#检查 `self.l1_reg` 是否为非空，即是否有L1正则化参数传递给这个对象。
            self.model.add_loss(self.l1_reg * self.l1_sparsity(out))  #  将L1正则化损失项添加到模型中    


        # self.model.build(input_shape=input_shape)

    def _build_cnn_model(self):#构建一个神经网络模型，其中包括卷积神经网络 (CNN) 层和长短时记忆网络 (LSTM) 层，用于处理时间序列数据。构建一个包括卷积神经网络和双向LSTM层的神经网络模型，用于处理时间序列数据。模型包括输入层、卷积层、全连接层、LSTM层、输出层和L1正则化损失。最终的输出是两部分的乘积，其中一部分来自全连接层，另一部分是由Sigmoid激活函数生成的掩码。
        tf.keras.backend.image_data_format() == 'channels_last'
        input_shape = (None, self.n_channels, 1)#定义了输入数据的形状。这是一个时间序列数据的形状，其中第一个维度是时间步数（可以是任何值，用`None`表示），第二个维度是通道数（`self.n_channels`），最后一个维度是通道深度为1。

        inputs = tf.keras.Input(shape=input_shape, name='Input_CNN')#这一行代码创建一个Keras输入层，定义了输入数据的形状和名称。
        fc = TimeDistributed(Conv1D(self.n_filters, self.n_channels, activation=self.activation_function, name="HL_D1"))(inputs)#构建了一个卷积神经网络层。`Conv1D` 表示一维卷积层，`self.n_filters` 是滤波器的数量，`self.n_channels` 是输入通道数，activation指定激活函数。`TimeDistributed`用于将卷积操作应用于每个时间步上的数据。
        fc = TimeDistributed(Flatten())(fc)#将卷积层的输出展平，以便将其输入到后续层中。
            
        # LSTM path
        lstm1 = Bidirectional(GRU(self.n_lstm_units, return_sequences=True), name='GRU')(fc)#构建了一个双向的长短时记忆网络 (LSTM) 层。`GRU` 是LSTM层的一种变体，`self.n_lstm_units` 是LSTM单元的数量，`return_sequences=True` 指定该层应返回整个时间序列而不仅仅是最后一个时间步的输出。`Bidirectional` 用于创建一个双向LSTM，它同时考虑了正向和反向的时间信息。 `lstm1` 是LSTM层的输出。
        mask = TimeDistributed(Dense(self.n_dipoles, activation="sigmoid"), name='Mask')(lstm1)#构建了一个时间分布式的全连接层，用于生成一个掩码（mask）。这个掩码的目的是产生权重，用于乘以另一个部分的输出。指定了该层的激活函数，通常用于生成0到1之间的权重。 `mask` 是掩码的输出。

        direct_out = TimeDistributed(Dense(self.n_dipoles, activation="tanh", name="Output_Final"))(fc)#构建了另一个时间分布式的全连接层，用于生成另一个输出。指定了该层的激活函数，通常用于生成在-1到1之间的输出。 `direct_out` 是该层的输出。
        multi = multiply([direct_out, mask], name="multiply")#将上面生成的两个部分（`direct_out` 和 `mask`）相乘，以生成最终的输出。

        self.model = tf.keras.Model(inputs=inputs, outputs=multi, name='CNN_Model')#创建了一个Keras模型对象 self.model，将输入层 inputs 和最终输出层 multi 组合在一起，给模型一个名称。
        if self.l1_reg is not None:#检查 self.l1_reg 是否为非空，即是否有L1正则化参数传递给这个对象。
            self.model.add_loss(self.l1_reg * self.l1_sparsity(multi))#将L1正则化损失项添加到模型中，L1正则化通常用于鼓励模型中的权重参数变得稀疏。
        # model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def _build_convdip_model(self):
        # self.model = keras.Sequential(name='ConvDip-model')
        tf.keras.backend.set_image_data_format('channels_last')#设置Keras的图像数据格式为'channels_last'，适用于卷积神经网络 (CNN) 处理，其中通道是数据的最后一个维度。
        input_shape = (None, *self.interp_channel_shape, 1)#定义了输入数据的形状。None 表示时间步数可以是任何值，self.interp_channel_shape 是通道形状，1 是通道深度。
        inputs = tf.keras.Input(shape=input_shape, name='Input_ConvDip')#创建一个Keras输入层，定义了输入数据的形状和名称。
        # Some definitions
              

        # Hidden Dense layer(s):构建了卷积神经网络 (CNN) 的一部分，包括卷积层和丢弃层。卷积层是用于从输入数据中提取特征的层。
        if not isinstance(self.n_dense_units, (tuple, list)):
            self.n_dense_units = [self.n_dense_units] * self.n_dense_layers
        
        if not isinstance(self.dropout, (tuple, list)):
            dropout = [self.dropout]*(self.n_dense_layers+self.n_lstm_layers)
        else:
            dropout = self.dropout

        # self.model.add(InputLayer(input_shape=input_shape, name='Input'))
        add_to = inputs
        for i in range(self.n_lstm_layers):
            conv = TimeDistributed(Conv2D(self.n_filters, self.kernel_size, activation=self.activation_function, name=f"Conv2D_{i}"))(add_to)
            conv = Dropout(dropout[i], name=f'Drop_conv2d_{i}')(conv)
            add_to = conv



        flat = TimeDistributed(Flatten())(conv)#将卷积层的输出展平，以便将其输入到全连接层中
        add_to = flat#将展平后的数据保存到 add_to 变量中，以便将其输入到全连接层中。
        for i in range(self.n_dense_layers):#构建了全连接层，用于对展平的数据进行前馈神经网络处理。
            dense = TimeDistributed(Dense(self.n_dense_units[i], activation=self.activation_function, name=f'FC_{i}'))(add_to)
            dense = Dropout(dropout[i], name=f'Drop_FC_{i}')(dense)#TimeDistributed 用于将全连接操作应用于每个时间步上的数据。dense 是当前全连接层的输出。
            add_to = dense

        # Outout Layer
        out = TimeDistributed(Dense(self.n_dipoles, activation='linear'), name='FC_Out')(dense)#构建了最终的输出层，输出的形状是 (None, self.n_dipoles)，激活函数是线性的。
        self.model = tf.keras.Model(inputs=inputs, outputs=out, name='ConvDip_Model')#创建了一个Keras模型对象 self.model，将输入层 inputs 和最终输出层 out 组合在一起，给模型一个名称
        if self.l1_reg is not None:#是否有L1正则化参数传递给这个对象。
            self.model.add_loss(self.l1_reg * self.l1_sparsity(out))
        

    @staticmethod
    def l1_sparsity(x):#用于计算L1稀疏性。它接受输入张量 `x`，首先对 `x` 进行L2范数归一化（L2标准化），然后计算其绝对值的平均值。这个方法似乎用于计算L1正则化的稀疏性项，以便将其添加到模型中。
        new_x = tf.math.l2_normalize(x)
        return K.mean(K.abs(new_x))
        
  

    def _freeze_lstm(self):#用于冻结模型中所有的LSTM（或RNN）层。它遍历模型的所有层，检查每个层的名称是否包含'LSTM'或'RNN'，如果包含，则将该层的 `trainable` 属性设置为 `False`，使其不可训练。
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name or 'RNN' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False
    
    def _unfreeze_lstm(self):#用于解冻之前冻结的LSTM（或RNN）层。它遍历模型的所有层，检查每个层的名称是否包含'LSTM'或'RNN'，如果包含，则将该层的 `trainable` 属性设置为 `True`，使其可训练。
        for i, layer in enumerate(self.model.layers):
            if 'LSTM' in layer.name or 'RNN' in layer.name:
                print(f'unfreezing {layer.name}')
                self.model.layers[i].trainable = True
    
    def _freeze_fc(self):#用于冻结模型中的全连接（FC）层，但不包括输出层。它遍历模型的所有层，检查每个层的名称是否包含'FC'但不包含'Out'，如果满足条件，则将该层的 `trainable` 属性设置为 `False`，使其不可训练。
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name and not 'Out' in layer.name:
                print(f'freezing {layer.name}')
                self.model.layers[i].trainable = False

    def _unfreeze_fc(self):#用于解冻之前冻结的全连接（FC）层，包括输出层。它遍历模型的所有层，检查每个层的名称是否包含'FC'，如果包含，则将该层的 `trainable` 属性设置为 `True`，使其可训练。
        for i, layer in enumerate(self.model.layers):
            if 'FC' in layer.name:
                print(f'unfreezing {layer.name}')
                self.model.layers[i].trainable = True

    def _build_perceptron_model(self):#构建一个神经网络模型，其中包括一个或多个全连接隐藏层和一个全连接输出层，适用于处理时间序列数据。
        ''' Build the artificial neural network model using Dense layers.用于构建一个基于全连接层（Dense层）的人工神经网络模型。
        '''
        input_shape = (None, None, self.n_channels)#前两个维度是`None`，表示可以接受可变长度的输入序列，而第三个维度是 `self.n_channels`，表示输入的通道数
        tf.keras.backend.set_image_data_format('channels_last')#确保模型适用于通道位于最后的数据。

        self.model = keras.Sequential()#创建一个Keras顺序模型对象 `self.model`，这是一个线性层次的神经网络模型。
        # Add hidden layers
        for _ in range(self.n_dense_layers):#开始一个循环，用于添加多个隐藏层。`self.n_dense_layers` 指定了要添加的隐藏层的数量。
            self.model.add(TimeDistributed(Dense(units=self.n_dense_units,
                                activation=self.activation_function)))#在模型中添加一个时间分布式的全连接层（TimeDistributed Dense层）。该层具有 `self.n_dense_units` 个神经元，指定的激活函数为 `self.activation_function`。`TimeDistributed` 用于将该层应用于每个时间步上的数据。
        # Add output layer
        self.model.add(TimeDistributed(Dense(self.n_dipoles, activation='linear')))#在模型中添加一个时间分布式的全连接输出层，用于生成模型的输出。该层有 `self.n_dipoles` 个神经元，激活函数为线性的。
        
        # Build model with input layer
        self.model.build(input_shape=input_shape)#用指定的输入形状 `input_shape` 构建模型，以确保模型与输入数据的形状兼容。

    


    def _check_model(self, eeg):
        ''' Check whether the current forward model has the same 
        channels as the eeg. Rebuild model if thats not the case. 这是一个类方法，用于检查当前前向模型是否与给定的EEG数据通道一致，并在需要时重新构建模型。
        
        Parameters
        ----------
        eeg : mne.Epochs or equivalent
            The EEG instance.

        '''
        # Dont do anything if model is already built.
        if self.compiled:#是否模型已经被编译（`self.compiled`为True）。如果模型已编译，就不执行任何操作，因为编译后的模型不能再次构建。
            return
        
        # Else assure that channels are appropriate
        if eeg[0].ch_names != self.fwd.ch_names:#检查给定的EEG数据 `eeg` 的通道名是否与当前前向模型 `self.fwd` 的通道名不一致。如果通道名不一致，意味着模型的通道不匹配输入数据的通道。
            self.fwd = self.fwd.pick_channels(eeg[0].ch_names)#如果通道名不一致，这一行代码将更新前向模型 `self.fwd`，只保留与给定EEG数据通道一致的通道。
            # Write all changes to the attributes
            self._embed_fwd(self.fwd)#调用 `_embed_fwd` 方法，以确保所有与前向模型相关的属性都得到正确的更新。
        
        self.n_timepoints = len(eeg[0].times)#获取给定EEG数据的时间点数量，将其存储在模型的 `self.n_timepoints` 属性中。
        # Finally, build model
        self._build_model()#调用 `_build_model` 方法，重新构建模型以适应新的通道结构，以确保与给定EEG数据一致。
        print("Net _check_model ")
            
    def scale_p(self, y_est, x_true):
        ''' Scale the prediction to yield same estimated GFP as true GFP，用于将估计的源向量 `y_est` 进行缩放，以使其具有与真实全局场电位 (GFP) 估计相同的标准差。确保估计的源向量具有与真实EEG数据相匹配的GFP，通过计算缩放因子并将估计的源向量缩放到所需的标准差水平。

        Parameters
        ---------
        y_est : numpy.ndarray，预测的源数据
            The estimated source vector.
        x_true : numpy.ndarray，原始的输入EEG数据
            The original input EEG vector.
        
        Return
        ------
        y_est_scaled : numpy.ndarray
            The scaled estimated source vector.
        
        '''
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:#检查估计的源向量 `y_est` 是否全为零。如果最大值等于零，表明 `y_est` 已经是零向量，无需进行缩放，直接返回原始向量。
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))#将输入的 `y_est` 和 `x_true` 转换为NumPy数组，并使用 `np.squeeze` 函数去除可能存在的多余的维度。
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)#通过矩阵乘法，使用前向模型（`self.leadfield`）将估计的源向量 `y_est` 转换为估计的EEG数据 `x_est
        gfp_true = np.std(x_true)#计算真实EEG数据 `x_true` 的全局场电位 (GFP) 的标准差。
        gfp_est = np.std(x_est)#计算估计的EEG数据 `x_est` 的全局场电位 (GFP) 的标准差。
        scaler = gfp_true / gfp_est#计算缩放因子 `scaler`，它是真实GFP与估计GFP之间的比率。这个比率用于将估计的源向量缩放到与真实GFP相匹配。
        y_est_scaled = y_est * scaler#将估计的源向量 `y_est` 乘以缩放因子 `scaler`，以获得已经缩放的估计源向量 `y_est_scaled`。
        print("Net scale_p ")
        return y_est_scaled#返回已经缩放的估计源向量 `y_est_scaled`，其标准差与真实GFP相匹配。
        
    def solve_p(self, y_est, x_true):
        '''一个实例方法，用于解决估计的源向量 `y_est`，以使其在与真实EEG数据 `x_true` 之间具有最佳相关性。
        Parameters
        ---------
        y_est : numpy.ndarray
            The estimated source vector.
        x_true : numpy.ndarray
            The original input EEG vector.
        
        Return
        ------
        y_scaled : numpy.ndarray
            The scaled estimated source vector.
        
        '''
        # Check if y_est is just zeros:
        if np.max(y_est) == 0:#检查估计的源向量 y_est是否全为零。如果最大值等于零，表明 `y_est` 已经是零向量，无需进行解决，直接返回原始向量。
            return y_est
        y_est = np.squeeze(np.array(y_est))
        x_true = np.squeeze(np.array(x_true))#将输入的 `y_est` 和 `x_true` 转换为NumPy数组，并使用 `np.squeeze` 函数去除可能存在的多余的维度。
        # Get EEG from predicted source using leadfield
        x_est = np.matmul(self.leadfield, y_est)#通过矩阵乘法，使用前向模型（`self.leadfield`）将估计的源向量 `y_est` 转换为估计的EEG数据 `x_est`。

        # optimize forward solution优化前向模型
        tol = 1e-9#`tol` 是缩写，代表容差（tolerance）。在优化问题中，容差通常表示接受的误差范围。`tol` 设置为1e-9，即一个非常小的数值，用于指定在优化过程中允许的目标函数的变化范围非常小。如果优化的目标函数的变化小于等于这个容差值，优化算法将停止。这有助于确保优化算法收敛到一个足够接近最优解的点。

        options = dict(maxiter=1000, disp=False)#指定了最大的迭代次数，即优化算法尝试寻找最优解的最大次数。在这里，它设置为1000，表示最多允许进行1000次迭代。如果在达到这个迭代次数之前目标函数没有足够收敛，优化算法将提前终止。

        # base scaling
        rms_est = np.mean(np.abs(x_est))
        rms_true = np.mean(np.abs(x_true))
        base_scaler = rms_true / rms_est#计算了一个基本缩放因子 `base_scaler`，该因子基于估计的EEG数据 `x_est` 和真实EEG数据 `x_true` 的均方根。

        
        opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
            bounds=(0, 1), method='bounded', options=options, tol=tol)#使用 `minimize_scalar` 函数来最小化 `correlation_criterion` 函数，以找到合适的缩放因子。 `correlation_criterion` 函数是一个用于计算估计的EEG数据和真实EEG数据之间相关性的函数。 `args` 参数传递给 `correlation_criterion` 函数，其中包括前向模型（`self.leadfield`）、估计的源向量 `y_est` 和缩放因子 `base_scaler`，以及真实EEG数据 `x_true`。 `bounds` 指定了缩放因子的取值范围为 `[0, 1]`。 `method` 指定了优化方法为 'bounded'，表示使用约束优化。`options` 包含了一些优化选项，如最大迭代次数和显示信息的设置。`tol` 是容差值，用于确定何时认为优化已经收敛。
        
        # opt = minimize_scalar(self.correlation_criterion, args=(self.leadfield, y_est* base_scaler, x_true), \
        #     bounds=(0, 1), method='L-BFGS-B', options=options, tol=tol)

        scaler = opt.x#从优化结果中获取缩放因子
        y_scaled = y_est * scaler * base_scaler#将估计的源向量 `y_est` 乘以缩放因子 `scaler` 和基本缩放因子 `base_scaler`，以获得已经缩放的估计源向量 `y_scaled`。
        print("Net solve_p")
        return y_scaled#返回已经缩放的估计源向量 `y_scaled`，以使其与真实EEG数据 `x_true` 具有最佳相关性

    @staticmethod
    def correlation_criterion(scaler, leadfield, y_est, x_true):
        ''' Perform forward projections of a source using the leadfield.
        This is the objective function which is minimized in Net::solve_p().
        这是一个静态方法，用作目标函数，将在solve_p方法中被最小化，以寻找最佳的缩放因子scaler。
  
        Parameters
        ----------
        scaler : float
            scales the source y_est
        leadfield : numpy.ndarray
            The leadfield (or sometimes called gain matrix).
        y_est : numpy.ndarray
            Estimated/predicted source.
        x_true : numpy.ndarray
            True, unscaled EEG.
        '''

        x_est = np.matmul(leadfield, y_est) #使用矩阵乘法，将估计的源向量 `y_est` 投影到前向模型（`leadfield`）中，以获得估计的EEG数据 `x_est`。
        error = np.abs(pearsonr(x_true-x_est, x_true)[0])#计算估计的EEG数据 `x_est` 与真实EEG数据 `x_true` 之间的相关性。具体来说，它使用 `pearsonr` 函数计算这两个信号的皮尔逊相关系数，并取绝对值。这个相关性值越接近1，表示估计的EEG数据与真实EEG数据之间的相关性越高，代表估计得越好。
        return error#方法返回相关性误差 `error`，这是一个衡量估计的源向量 `y_est` 与真实EEG数据 `x_true` 之间相关性的度量。最小化此误差将帮助找到最佳的缩放因子 `scaler`
    
    def save(self, path, name='model'):#这是一个实例方法，用于保存模型和相关的数据到指定的文件路径。
        # get list of folders in path，将模型和相关数据保存到指定的文件夹中，并允许多个版本的模型在同一目录下共存。同时，它还提供了一种方式来保存和加载模型实例，以便后续的使用。
        list_of_folders = os.listdir(path)#获取指定路径 `path` 下的文件夹列表，并将其存储在 `list_of_folders` 变量中。
        model_ints = []#创建一个空列表 `model_ints`，用于存储已经存在的模型文件夹中的整数编号。
        for folder in list_of_folders:#遍历文件夹列表
            full_path = os.path.join(path, folder)#构建文件夹的完整路径，以便后续检查和确定模型名称是否已存在。
            if not os.path.isdir(full_path):#码检查当前路径是否为文件夹，如果不是文件夹则跳过该路径的处理，继续下一个路径。
                continue
            if folder.startswith(name):#检查文件夹名是否以指定的 `name` 开头，以确定是否是模型文件夹。
                new_integer = int(folder.split('_')[-1])#从文件夹名中提取整数编号。假设模型文件夹的命名规则是 `name_0`、`name_1`、`name_2` 等，其中整数部分表示模型的版本。
                model_ints.append(new_integer)#将提取的整数编号添加到 `model_ints` 列表中。
        if len(model_ints) == 0:#如果没有找到任何现有模型文件夹，则创建一个新的模型版本，编号从0开始。
            model_name = f'\\{name}_0'
        else:
            model_name = f'\\{name}_{max(model_ints)+1}'
        new_path = path+model_name#构建新模型文件夹的完整路径。
        os.mkdir(new_path)#创建新的模型文件夹

        # Save model only
        self.model.save(new_path)#将当前模型保存到新创建的模型文件夹中。
        # self.model.save_weights(new_path)

        # copy_model = tf.keras.models.clone_model(self.model)
        # copy_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.35), loss='huber')
        # copy_model.set_weights(self.model.get_weights())        
        # Save rest
        # Delete model since it is not serializable
        self.model = None#将当前模型设置为`None`，因为模型对象不能直接序列化保存，所以需要分开保存模型和模型的实例。

        with open(new_path + '\\instance.pkl', 'wb') as f:#使用 `pickle` 库将当前模型实例序列化并保存到文件中
            pkl.dump(self, f)
        
        # Attach model again now that everything is saved
        try:
            self.model = tf.keras.models.load_model(new_path, custom_objects={'loss': self.loss})
        except:
            print("Load model did not work using custom_objects. Now trying it without...")
            self.model = tf.keras.models.load_model(new_path)
        
        return self#方法返回更新后的模型实例。

    @staticmethod
    def make_interpolator(elec_pos, res=9, ch_type='eeg'):#用于创建一个插值器（interpolator）对象，用于在电极位置之间执行插值操作。
        extrapolate = _check_extrapolate('auto', ch_type)
        sphere = sphere = _check_sphere(None)#调用 `_check_sphere` 函数来获取球面（sphere）的几何信息，通常表示电极放置在球体上。
        outlines = 'head'#用于指定头部的轮廓。默认设置为 `'head'`，表示使用头部的轮廓信息。
        outlines = _make_head_outlines(sphere, elec_pos, outlines, (0., 0.))#调用 `_make_head_outlines` 函数，根据球面、电极位置、轮廓信息和球心位置来生成头部的轮廓信息。
        border = 'mean'#用于指定插值边界的处理方式。默认设置为 `'mean'`，表示使用平均值。
        extent, Xi, Yi, interpolator = _setup_interp(
            elec_pos, res, extrapolate, sphere, outlines, border)#调用 `_setup_interp` 函数，用于设置插值的一些参数，包括电极位置、分辨率、外推选项、球面信息、轮廓信息和边界处理方式。
        interpolator.set_locations(Xi, Yi)#使用插值器对象的 `set_locations` 方法来设置插值点的位置，其中 `Xi` 和 `Yi` 是要插值的坐标点。

        return interpolator#返回创建的插值器对象 `interpolator`。

    

def build_nas_lstm(hp):
    ''' Find optimal model using keras tuner.使用 Keras Tuner 构建一个具有可调参数的神经网络模型。通过 Keras Tuner 创建一个具有可调参数的 LSTM 模型，以进行超参数搜索和模型优化。超参数包括层数、单元数、丢失率、激活函数、优化器、学习率等，可以通过 `hp` 参数来进行调整。这有助于寻找最佳的超参数配置以获得最佳的模型性能。这段代码的设计原理是为了提供一个可自动搜索最佳超参数配置的模型构建过程。
    '''
    n_dipoles = 1284#源的数量
    n_channels = 64#通道数
    n_lstm_layers = hp.Int("lstm_layers", min_value=0, max_value=3, step=1)#定义了模型中 LSTM 层的数量，这些参数可以在超参数搜索时进行调整。
    n_dense_layers = hp.Int("dense_layers", min_value=0, max_value=3, step=1)#定义了模型中全连接层的数量，这些参数可以在超参数搜索时进行调整。
    activation_out = 'linear'  # hp.Choice(f"activation_out", ["tanh", 'sigmoid', 'linear'])定义输出层的激活函数
    activation = 'relu'  # hp.Choice('actvation_all', all_acts)定义中间层的激活函数

    model = keras.Sequential(name='LSTM_NAS')#创建了一个 Keras 序贯模型，并设置了模型的名称。
    tf.keras.backend.set_image_data_format('channels_last')#设置 Keras 后端的图像数据格式为 `'channels_last'`，表示通道数据在最后的维度。
    input_shape = (None, n_channels)#定义了输入数据的形状，其中 `None` 表示可变长度序列，`n_channels` 表示通道数量。
    model.add(InputLayer(input_shape=input_shape, name='Input'))

    # LSTM layers
    for i in range(n_lstm_layers):#通过循环创建了指定数量的 LSTM 层，每个 LSTM 层具有不同的超参数，如单元数（`n_lstm_units`）和丢失率（`dropout`）。
        n_lstm_units = hp.Int(f"lstm_units_l-{i}", min_value=25, max_value=500, step=1)
        dropout = hp.Float(f"dropout_lstm_l-{i}", min_value=0, max_value=0.5)
        model.add(Bidirectional(LSTM(n_lstm_units, 
            return_sequences=True, input_shape=input_shape, 
            dropout=dropout, activation=activation), 
            name=f'LSTM{i}'))
    # Hidden Dense layer(s):
    for i in range(n_dense_layers):#通过循环创建了指定数量的全连接层，每个全连接层具有不同的超参数，如单元数（`n_dense_units`）和丢失率（`dropout`）。
        n_dense_units = hp.Int(f"dense_units_l-{i}", min_value=50, max_value=1000, step=1)
        dropout = hp.Float(f"dropout_dense_l-{i}", min_value=0, max_value=0.5)

        model.add(TimeDistributed(Dense(n_dense_units, 
            activation=activation), name=f'FC_{i}'))
        model.add(Dropout(dropout, name=f'DropoutLayer_dense_{i}'))

    # Final For-each layer:创建一个最终的全连接输出层，用于生成模型的输出。
    model.add(TimeDistributed(
        Dense(n_dipoles, activation=activation_out), name='FC_Out')
    )
    model.build(input_shape=input_shape)
    momentum = hp.Float('Momentum', min_value=0, max_value=0.9)
    nesterov = hp.Choice('Nesterov', [False, True])#定义了优化器的超参数，包括动量（`momentum`）和是否使用 Nesterov 加速梯度下降（`nesterov`）
    learning_rate = hp.Choice('learning_rate', [0.01, 0.001])#学习率可以选择为 0.01 或 0.001
    optimizer = hp.Choice("Optimizer", [0,1,2])#优化器可以选择为 RMSprop、Adam 或带有 Nesterov 的 SGD。
    optimizers = [keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum), keras.optimizers.Adam(learning_rate=learning_rate), keras.optimizers.SGD(learning_rate=learning_rate, nesterov=nesterov)]#创建优化器列表 `optimizers`，并根据选择的优化器超参数设置模型的优化器。

    model.compile(#这一行代码编译模型，设置了模型的优化器、损失函数（"huber"）以及评估指标。这些超参数都是通过超参数搜索来选择的。
        optimizer=optimizers[optimizer],
        loss="huber",
        # metrics=[tf.keras.metrics.AUC()],
        # metrics=[evaluate.modified_auc_metric()],
        metrics=[evaluate.auc],
    )
    print("net build_nas_lstm")
    return model#函数返回创建的神经网络模型。

 #class EnsembleNet:
#     ''' Uses ensemble of neural networks to perform predictions,类的整体目的是使用神经网络集成进行预测。
#     Attributes
#     ----------
#     nets : list，神经网络实例的列表，用于存储集成中的不同神经网络。
#         a list of instances of the Net class
#     ensemble_mode : str，字符串，决定如何组合不同预测的模式。在这里，它描述了一种模式，即 'average'，表示通过平均所有预测来得到最终结果。
#         Decides how the various predictions will be combined.
#         'average' : average all predictions with equal weight
    
#     Methods
#     -------
#     predict : performs predictions with each Net instance and combines them.，predict 方法：用于执行每个神经网络实例的预测并将它们组合在一起。
#     vote_average : the implementation of the ensemble_mode 'average'，vote_average 方法：实现了 'average' 模式的投票策略，用于将不同预测组合成最终的集成预测。

#     Examples
#     --------
#     ### Build two Nets nad train them
#     k = 2  # number of models
#     nets = [Net(fwd).fit(simulation.eeg_data, simulation.source_data) for _ in range(k)]
#     ### Combine them into an EnsembleNet
#     ens_net = EnsembleNet(nets)
#     ### Perform prediction
#     y_hat = nets[0].predict(simulation_test)
#     y_hat_ens = ens_net.predict(simulation_test.eeg_data)
#     ### Plot result
#     a = simulation_test.source_data.plot(**plot_params)  # Ground truth
#     b = y_hat.plot(**plot_params)  # single-model prediction
#     c = y_hat_ens.plot(**plot_params)  # ensemble predicion



#     '''
#     def __init__(self, nets, ensemble_mode='average'):#__init__ 方法是类的构造函数。它接受神经网络实例的列表 nets 和一个用于决定如何组合各个预测的 ensemble_mode 参数。如果 ensemble_mode 是 'average'，则设置 self.vote 为 self.vote_average 方法；否则，引发属性错误。
#         self.nets = nets
#         self.ensemble_mode = ensemble_mode
        
#         if ensemble_mode == 'average':
#             self.vote = self.vote_average
#         # if ensemble_mode == 'stack':
#         #     self.vote = self.vote_stack
#         else:
#             msg = f'ensemble_mode {ensemble_mode} not supported'
#             raise AttributeError(msg)
        

#     def predict(self, *args):#predict 方法接受一些参数，对每个神经网络实例调用其 predict 方法进行预测。然后，将所有预测的数据堆叠在一起，通过 self.vote 方法进行集成。最终，创建一个新的预测对象并返回。
#         predictions = [net.predict(args[1]) for net in self.nets]
#         predictions_data = np.stack([prediction.data for prediction in predictions], axis=0)
        
#         ensemble_prediction = predictions[0]
#         ensemble_prediction.data = self.vote(predictions_data)

#         return ensemble_prediction

#     def vote_average(self, predictions_data):#vote_average 方法是一种投票策略，对于 'average' 模式，它计算各个预测的平均值，作为最终的集成预测
#         return np.mean(predictions_data, axis=0)

# class BoostNet:
#     ''' The Boosted neural network class that creates and trains the boosted model. 
        
#     Attributes
#     ----------
#     fwd : mne.Forward
#         the mne.Forward forward model class.
#     n_nets : int
#         The number of neural networks to use.
#     n_layers : int
#         Number of hidden layers in the neural network.
#     n_neurons : int
#         Number of neurons per hidden layer.
#     activation_function : str
#         The activation function used for each fully connected layer.

#     Methods
#     -------
#     fit : trains the neural network with the EEG and source data
#     train : trains the neural network with the EEG and source data
#     predict : perform prediciton on EEG data
#     evaluate : evaluate the performance of the model
#     '''

#     def __init__(self, fwd, n_nets=5, n_layers=1, n_neurons=128, 
#         activation_function='swish', verbose=False):

#         self.nets = [Net(fwd, n_layers=n_layers, n_neurons=n_neurons, 
#             activation_function=activation_function, verbose=verbose) 
#             for _ in range(n_nets)]

#         self.linear_regressor = linear_model.LinearRegression()

#         self.verbose=verbose
#         self.n_nets = n_nets

#     def fit(self, *args, **kwargs):
#         ''' Train the boost model.

#         Parameters
#         ----------
#         *args : esinet.simulation.Simulation
#             Can be either 
#                 eeg : mne.Epochs/ numpy.ndarray
#                     The simulated EEG data
#                 sources : mne.SourceEstimates/ list of mne.SourceEstimates
#                     The simulated EEG data
#                 or
#                 simulation : esinet.simulation.Simulation
#                     The Simulation object

#         **kwargs
#             Arbitrary keyword arguments.

#         Return
#         ------
#         self : BoostNet()
#         '''

#         eeg, sources = self._handle_data_input(args)
#         self.subject = sources.subject if type(sources) == mne.SourceEstimate else sources[0].subject

#         if self.verbose:
#             print("Fit neural networks")
#         self._fit_nets(eeg, sources, **kwargs)

#         ensemble_predictions, _ = self._get_ensemble_predictions(eeg, sources)
           
#         if self.verbose:
#             print("Fit regressor")
#         # Train linear regressor to combine predictions
#         self.linear_regressor.fit(ensemble_predictions, sources.data.T)

#         return self
    
#     def predict(self, *args):
#         ''' Perform prediction of sources based on EEG data using the Boosted Model.
        
#         Parameters
#         ----------
#         *args : 
#             Can be either 
#                 eeg : mne.Epochs/ numpy.ndarray
#                     The simulated EEG data
#                 sources : mne.SourceEstimates/ list of mne.SourceEstimates
#                     The simulated EEG data
#                 or
#                 simulation : esinet.simulation.Simulation
#                     The Simulation object
#         **kwargs
#             Arbitrary keyword arguments.
        
#         Return
#         ------
#         '''

#         eeg, sources = self._handle_data_input(args)

#         ensemble_predictions, y_hats = self._get_ensemble_predictions(eeg, sources)
#         prediction = np.clip(self.linear_regressor.predict(ensemble_predictions), a_min=0, a_max=np.inf)
        
#         y_hat = y_hats[0]
#         y_hat.data = prediction.T
#         return y_hat

#     def evaluate_mse(self, *args):
#         ''' Evaluate the model regarding mean squared error
        
#         Parameters
#         ----------
#         *args : 
#             Can be either 
#                 eeg : mne.Epochs/ numpy.ndarray
#                     The simulated EEG data
#                 sources : mne.SourceEstimates/ list of mne.SourceEstimates
#                     The simulated EEG data
#                 or
#                 simulation : esinet.simulation.Simulation
#                     The Simulation object

#         Return
#         ------
#         mean_squared_errors : numpy.ndarray
#             The mean squared error of each sample

#         Example
#         -------
#         net = BoostNet()
#         net.fit(simulation)
#         mean_squared_errors = net.evaluate(simulation)
#         print(mean_squared_errors.mean())
#         '''

#         eeg, sources = self._handle_data_input(args)
#         y_hat = self.predict(eeg, sources).data
#         y_true = sources.data
#         mean_squared_errors = np.mean((y_hat - y_true)**2, axis=0)
#         return mean_squared_errors


#     def _get_ensemble_predictions(self, *args):

#         eeg, sources = self._handle_data_input(args)

#         y_hats = [subnet.predict(eeg, sources) for subnet in self.nets]
#         ensemble_predictions = np.stack([y_hat[0].data for y_hat in y_hats], axis=0).T
#         ensemble_predictions = ensemble_predictions.reshape(ensemble_predictions.shape[0], np.prod((ensemble_predictions.shape[1], ensemble_predictions.shape[2])))
#         return ensemble_predictions, y_hats

#     def _fit_nets(self, *args, **kwargs):

#         eeg, sources = self._handle_data_input(args)
#         n_samples = eeg.get_data().shape[0]
#         # sample_weight = np.ones((sources._data.shape[1]))
        
#         for net in self.nets:
#             sample_idc = np.random.choice(np.arange(n_samples), 
#                 int(0.8*n_samples), replace=True)
#             eeg_bootstrap = eeg.copy()[sample_idc]
#             sources_bootstrap = sources.copy()
#             sources_bootstrap.data = sources_bootstrap.data[:, sample_idc]
#             net.fit(eeg_bootstrap, sources_bootstrap, **kwargs)#, sample_weight=sample_weight)
#             # sample_weight = net.evaluate_mse(eeg, sources)
#             # print(f'new sample weights: mean={sample_weight.mean()} +- {sample_weight.std()}')

        
#     def _handle_data_input(self, arguments):
#         ''' Handles data input to the functions fit() and predict().
        
#         Parameters
#         ----------
#         arguments : tuple
#             The input arguments to fit and predict which contain data.
        
#         Return
#         ------
#         eeg : mne.Epochs
#             The M/EEG data.
#         sources : mne.SourceEstimates/list
#             The source data.

#         '''
#         if len(arguments) == 1:
#             if isinstance(arguments[0], (mne.Epochs, mne.Evoked, mne.io.Raw, mne.EpochsArray, mne.EvokedArray, mne.epochs.EpochsFIF)):
#                 eeg = arguments[0]
#                 sources = None
#             else:
#                 simulation = arguments[0]
#                 eeg = simulation.eeg_data
#                 sources = simulation.source_data
#                 # msg = f'First input should be of type simulation or Epochs, but {arguments[1]} is {type(arguments[1])}'
#                 # raise AttributeError(msg)

#         elif len(arguments) == 2:
#             eeg = arguments[0]
#             sources = arguments[1]
#         else:
#             msg = f'Input is {type()} must be either the EEG data and Source data or the Simulation object.'
#             raise AttributeError(msg)

#         return eeg, sources
