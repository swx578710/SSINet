import numpy as np#NumPy（用于数值计算）
import matplotlib.pyplot as plt#Matplotlib（用于绘图）
from sklearn.metrics import auc, roc_curve#用于处理ROC曲线相关的指标
import tensorflow as tf
from tensorflow.keras import backend as K#用于深度学习框架
from scipy.spatial.distance import cdist#用于计算源之间的距离
from copy import deepcopy#用于创建深拷贝的数据结构
import time#用于计时

def get_maxima_mask(y, pos, k_neighbors=5, threshold=0.1, min_dist=30,
    argsorted_distance_matrix=None):#计算一组源的二进制掩码，用于识别并标记重要的源最大值
    ''' Returns the mask containing the source maxima (binary).
     `get_maxima_mask` 是一个函数，接受多个参数，用于确定源的最大值并生成二进制掩码，指示哪些源是显著的。
    Parameters
    ----------
    y : numpy.ndarray，源
        The source
    pos : numpy.ndarray，源的位置矩阵
        The dipole position matrix
    k_neighbors : int，考虑的近邻数
        The number of neighbors to incorporate for finding maximum
    threshold : float，最大值的阈值，以确定哪些最大值是显著的
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist：源之间的最小距离，以排除接近的最大值
    argsorted_distance_matrix：已经计算好的源之间距离的排序矩阵
    '''
    if argsorted_distance_matrix is None:#如果未提供 `argsorted_distance_matrix` 参数，则计算源之间的距离，并使用 `np.argsort` 对这些距离进行排序以创建 `argsorted_distance_matrix`，该矩阵将用于查找最近邻源。
        argsorted_distance_matrix = np.argsort(cdist(pos, pos), axis=1)

    
    y = np.abs(y)#计算源 `y` 的绝对值，以便后续比较。
    threshold = threshold*np.max(y)#计算 `threshold` 的实际值，将其设置为 `threshold` 与 `y` 中最大值的乘积。这个值将作为控制源显著性的阈值。
    #t_start = time.time()#记录时间以供性能分析。
    
    #print("argsorted_distance_matrix shape:", argsorted_distance_matrix.shape)
    #print("max value in argsorted_distance_matrix:", np.max(argsorted_distance_matrix))
    #print("min value in argsorted_distance_matrix:", np.min(argsorted_distance_matrix))
    
    # find maxima that surpass the threshold:
    close_idc = argsorted_distance_matrix[:, 1:k_neighbors+1]#创建一个 `close_idc` 数组，其中包含每个源的 `k_neighbors` 个最近邻源的索引。这些最近邻用于查找局部最大值。
    mask = ((y >= np.max(y[close_idc], axis=1)) & (y > threshold)).astype(int)#创建一个 `mask` 数组，初始化为与 `y` 相同长度的零数组。`mask` 将用于标记源是否是局部最大值。条件 `(y >= np.max(y[close_idc], axis=1))` 确保了一个源大于它的所有最近邻源，而 `(y > threshold)` 确保了源的值大于阈值。最终，`mask` 是一个二进制掩码，标识出局部最大值。
    
    # OLD CODE
    # mask = np.zeros((len(y)))
    # for i, _ in enumerate(y):
    #     distances = distance_matrix[i]
    #     close_idc = np.argsort(distances)[1:k_neighbors+1]
    #     if y[i] > np.max(y[close_idc]) and y[i] > threshold:
    #         mask[i] = 1
    
    #t_loop1 = time.time()# 记录一个时间点以供性能分析。
    # filter maxima
    maxima = np.where(mask==1)[0]#创建 `maxima` 数组，包含 `mask` 中值为1的源的索引。这些源被认为是最大值。
    distance_matrix_maxima = cdist(pos[maxima], pos[maxima])#.计算 `maxima` 之间的距离，以创建 `distance_matrix_maxima`，表示最大值之间的距离。
    for i, _ in enumerate(maxima):#遍历每个最大值，对于每个最大值，计算其与其他最大值之间的距离，并将距离小于min_dist的最大值标记为要删除的最大值。
        distances_maxima = distance_matrix_maxima[i]
        close_maxima = maxima[np.where(distances_maxima < min_dist)[0]]
        # If there is a larger maximum in the close vicinity->delete maximum
        if np.max(y[close_maxima]) > y[maxima[i]]:
            mask[maxima[i]] = 0
    #t_loop2 = time.time()
    #print("evaluate get_maxima_mask")
    return mask#返回包含最大值的二进制掩码 `mask`。这个掩码用于标记那些在局部区域内是显著的源最大值，并且不太接近其他最大值的源。
    
def get_maxima_pos(mask, pos):#返回局部最大源的三维坐标
    ''' Returns the positions of the maxima within mask.
    该函数接受两个参数，`mask` 和 `pos`，分别表示源的掩码和源的位置矩阵。它的作用是返回标记在 `mask` 中的源的位置。
    Parameters
    ----------
    mask : numpy.ndarray
        The source mask，源的掩码，`mask` 是一个包含二进制值的数组，指示哪些源是局部最大值。
    pos : numpy.ndarray
        The dipole position matrix，源的位置矩阵，pos` 是一个包含源位置的矩阵，其中每行表示一个源的位置坐标
    '''
    #print("evaluate get_maxima_pos")
    return pos[np.where(mask==1)[0]]#函数通过使用 `np.where(mask==1)` 来找到在掩码中标记为1的源的索引，然后返回这些源的位置坐标。

def eval_residual_variance(M_true, M_est):#计算脑电之间的残差方差
    ''' 
 该函数接受两个参数，`M_true` 和 `M_est`，分别表示真实的 EEG 数据和估计的 EEG 数据。它的作用是计算残差方差，即真实 EEG 数据和估计 EEG 数据之间的差异。
    Calculate the Residual Variance (1- goodness of fit) between the
    estimated EEG and the original EEG.
    
    Parameters
    ----------
    M_true : numpy.ndarray，真实的 EEG 数据，可以是单个时间点或时空数据。
        The true EEG data (as recorded). May be a single time point or
        spatio-temporal.
    M_est : numpy.ndarray，估计的 EEG 数据，是从估计的源投影而来的，也可以是单个时间点或时空数据。
        The estimated EEG data (projected from the estimated source). May be a
        single time point or spatio-temporal.
    计算残差方差：计算结果乘以100以获得百分比形式。函数返回一个百分比值，表示估计的 EEG 数据与真实 EEG 数据之间的拟合度。较低的残差方差表示估计结果与真实数据的拟合度较高，而较高的残差方差表示拟合度较低。
    
    '''
    #print("evaluate eval_residual_variance")
    return 100 *  np.sum( (M_true-M_est)**2 ) / np.sum(M_true**2)


def eval_mean_localization_error(y_true, y_est, pos, k_neighbors=5, 
    min_dist=30, threshold=0.2, ghost_thresh=40, argsorted_distance_matrix=None):
    ''' Calculate the mean localization error for an arbitrary number of 
    sources函数的主要目的是计算源的平均定位误差，可以适用于任意数量的源。
    
    Parameters
    ----------
    y_true : numpy.ndarray，表示真实源向量。它是一个NumPy数组（1D），包含了真实源的信息。
        The true source vector (1D)
    y_est : numpy.ndarray，表示估计源向量。它也是一个NumPy数组（1D），包含了估计的源的信息。
        The estimated source vector (1D)
    pos : numpy.ndarray，表示偶极子位置矩阵。它是一个NumPy数组，包含了每个源的位置坐标。
        The dipole position matrix
    k_neighbors : int，一个整数参数，表示用于查找最大值的邻居数目。
        The number of neighbors to incorporate for finding maximum
    threshold : float，一个浮点数参数，表示阈值，用于确定最大值的有效性。它是一个介于0和1之间的比例，例如，0.1表示绝对最大值的10%。
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int，表示最小有效距离（以毫米为单位），用于过滤最大值。较高的值会导致更多的最大值被过滤掉。
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    ghost_thresh : float/int，表示真实源和估计源之间的阈值距离。如果距离超过这个阈值，估计源将被标记为“ghost_source”（幽灵源）。
        The threshold distance between a true and a predicted source to not 
        belong together anymore. Predicted sources that have no true source 
        within the vicinity defined be ghost_thresh will be labeled 
        ghost_source.
    
    Return
    ------
    mean_localization_error : float，说明了函数返回的是一个浮点数，表示平均定位误差。
        The mean localization error between all sources in y_true and the 
        closest matches in y_est.
    '''
    if y_est.sum() == 0 or y_true.sum() == 0:#检查 `y_true` 和 `y_est` 是否都为零，如果是，返回 `nan`
        return np.nan
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)#对 `y_true` 和 `y_est` 进行深复制，以防止修改原始数据。
    if argsorted_distance_matrix is None:
        argsorted_distance_matrix = cdist(y_true.reshape(-1, 1), y_est.reshape(-1, 1)).argsort(axis=1)
    
    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist, 
        argsorted_distance_matrix=argsorted_distance_matrix), pos)#使用 `get_maxima_mask` 函数获取真实源掩码
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist, 
        argsorted_distance_matrix=argsorted_distance_matrix), pos)#使用 `get_maxima_mask` 函数获取估计源的掩码

    # Distance matrix between every true and estimated maximum，计算真实源和估计源之间的距离矩阵 `distance_matrix`。
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:，找到每个真实源的最近匹配估计源，并保存到 `closest_matches` 中
    closest_matches = distance_matrix.min(axis=1)
    # Filter for ghost sources， 过滤掉距离超过 `ghost_thresh` 的匹配
    closest_matches = closest_matches[closest_matches<ghost_thresh]
    
    # No source left -> return nan，如果没有匹配的源剩下，返回 `nan`
    if len(closest_matches) == 0:
        return np.nan
    mean_localization_error = np.mean(closest_matches)#计算 `closest_matches` 的平均值，作为平均定位误差，并返回该值。
    #print("evaluate eval_mean_localization_error")
    return mean_localization_error#返回平均定位误差

def eval_ghost_sources(y_true, y_est, pos, k_neighbors=5, 
    min_dist=30, threshold=0.2, ghost_thresh=40):
    ''' Calculate the number of ghost sources in the estimated source.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    
    Return
    ------
    n_ghost_sources : int
        The number of ghost sources.
    '''
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)#创建了输入数组的深层副本，以确保不会修改原始数据
    maxima_true = get_maxima_pos(#获取了真实源局部最大值的位置
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist), pos)
    maxima_est = get_maxima_pos(#获取了估计源的局部最大值的位置
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist), pos)
    
    # Distance matrix between every true and estimated maximum，创建了一个距离矩阵，表示每个真实源和估计源之间的距离
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:找到每个真实源的最近估计源。
    closest_matches = distance_matrix.min(axis=1)

    # Filter ghost sources，将超过幽灵阈值的距离的估计源筛选出来，这些源被认为是幽灵源。
    ghost_sources = closest_matches[closest_matches>=ghost_thresh]
    n_ghost_sources = len(ghost_sources)#计算幽灵源的数量
    #print("evaluate eval_ghost_sources")
    return n_ghost_sources#返回幽灵源的数量

def eval_found_sources(y_true, y_est, pos, k_neighbors=5, 
    min_dist=30, threshold=0.2, ghost_thresh=40):
    ''' Calculate the number of found sources in the estimated source.
    评估在估计的源中找到的真实源的数量。它采用真实源和估计源的一维向量，以及对应的位置信息，通过计算最近邻距离来确定在估计源中找到的真实源的数量。
    Parameters
    ----------
    y_true : numpy.ndarray
        The true source vector (1D)
    y_est : numpy.ndarray
        The estimated source vector (1D)
    pos : numpy.ndarray
        The dipole position matrix
    k_neighbors : int
        The number of neighbors to incorporate for finding maximum
    threshold : float
        Proportion between 0 and 1. Defined the minimum value for a maximum to 
        be of significance. 0.1 -> 10% of the absolute maximum
    min_dist : float/int
        The minimum viable distance in mm between maxima. The higher this 
        value, the more maxima will be filtered out.
    
    Return
    ------
    n_found_sources : int
        The Number of true found sources.
    '''
    y_true = deepcopy(y_true)
    y_est = deepcopy(y_est)
    maxima_true = get_maxima_pos(
        get_maxima_mask(y_true, pos, k_neighbors=k_neighbors, 
        threshold=threshold, min_dist=min_dist), pos)
    maxima_est = get_maxima_pos(
        get_maxima_mask(y_est, pos, k_neighbors=k_neighbors,
        threshold=threshold, min_dist=min_dist), pos)
    
    # Distance matrix between every true and estimated maximum
    distance_matrix = cdist(maxima_true, maxima_est)
    # For each true source find the closest predicted source:
    closest_matches = distance_matrix.min(axis=1)

    # Filter ghost sources，将距离小于 ghost_thresh 的估计源筛选出来，这些源被认为是真实找到的源
    found_sources = closest_matches[closest_matches<ghost_thresh]
    n_found_sources = len(found_sources)#计算真实找到的源的数量
    #print("evaluate  n_found_sources")
    return n_found_sources



def eval_mse(y_true, y_est):
    '''Returns the mean squared error between predicted and true source. 
    计算真实源和预测源之间的均方误差MSE，即它计算每个对应元素之间的平方差的均值。MSE是一种衡量预测与实际观测之间差异的指标，数值越小表示预测越准确。
    ''' 
    #print("evaluate eval_mse")
    return np.mean((y_true-y_est)**2)

def eval_nmse(y_true, y_est):
    '''Returns the normalized mean squared error between predicted and true source.
    此函数计算标准化的均方误差（NMSE）。首先，通过将源向量除以其绝对值的最大值，将源向量标准化为范围在[-1, 1]之间。然后，计算标准化源向量之间的均方误差。标准化可以使得在不同尺度的情况下，对误差的衡量更为公平。
    '''
    
    y_true_normed = y_true / np.max(np.abs(y_true))
    y_est_normed = y_est / np.max(np.abs(y_est))
    #print("evaluate eval_nmse")
    return np.mean((y_true_normed-y_est_normed)**2)

def eval_auc(y_true, y_est, pos, n_redraw=25, epsilon=0.25, 
    plot_me=False):
    ''' Returns the area under the curve metric between true and predicted source. 
该函数的目的是计算真实源和预测源之间的ROC曲线下的面积，分别对于靠近源和远离源的偶极子。ROC曲线是二分类问题中常用的性能评估指标，表示真阳性率（True Positive Rate）与假阳性率（False Positive Rate）之间的权衡关系。 AUC值越高，说明模型性能越好。
    Parameters
    ----------
    y_true : numpy.ndarray
        True source vector ，:numpy数组，表示真实源向量。
    y_est : numpy.ndarray
        Estimated source vector ，numpy数组，表示预测源向量
    pos : numpy.ndarray
        Dipole positions (points x dims)， numpy数组，表示偶极子的位置矩阵，形状为 (points x dims)
    n_redraw : int
        Defines how often the negative samples are redrawn.，整数，定义负样本被重新抽样的次数。
    epsilon : float
        Defines threshold on which sources are considered
        active.浮点数，定义哪些源被认为是活跃的阈值。
    plot_me,布尔值，如果为True，则绘制ROC曲线。
    Return
    ------
    auc_close : float
        Area under the curve for dipoles close to source. 浮点数，表示靠近源的偶极子的AUC值。
    auc_far : float
        Area under the curve for dipoles far from source. 浮点数，表示远离源的偶极子的AUC值。
    '''
    # Copy
    # t_start = time.time()
    if y_est.sum() == 0 or y_true.sum() == 0:
        return np.nan, np.nan#如果真实源或预测源中没有任何活跃的源，则返回两个NaN值。
    y_true = deepcopy(y_true)#深拷贝真实源向量，以防止对原始数据的更改。
    y_est = deepcopy(y_est)#深拷贝预测源向量，以防止对原始数据的更改。
    # Absolute values
    y_true = np.abs(y_true)#将真实源向量取绝对值。
    y_est = np.abs(y_est)#将预测源向量取绝对值。

    # Normalize values
    y_true /= np.max(y_true)#将真实源向量归一化为其最大值。
    y_est /= np.max(y_est)#将预测源向量归一化为其最大值。

    auc_close = np.zeros((n_redraw))#创建用于存储每次重新抽样后的AUC值的数组（close情况）
    auc_far = np.zeros((n_redraw))#创建用于存储每次重新抽样后的AUC值的数组（far情况）。
    
    # t_prep = time.time()
    # print(f'\tprep took {1000*(t_prep-t_start):.1f} ms')
    
    source_mask = (y_true>epsilon).astype(int)# 创建一个二值掩码，指示哪些源被认为是活跃的，基于真实源向量。

    numberOfActiveSources = int(np.sum(source_mask))#计算活跃源的数量。
    #print('numberOfActiveSources: ', numberOfActiveSources)
    numberOfDipoles = pos.shape[0]#获取位置矩阵中的总偶极子数。
    # Draw from the 20% of closest dipoles to sources (~100)
    closeSplit = int(round(numberOfDipoles / 5))#计算近源偶极子数量的拆分点。最近的20%
    # Draw from the 50% of furthest dipoles to sources
    farSplit = int(round(numberOfDipoles / 2))#计算远源偶极子数量的拆分点。最远的50%
    # t_prep = time.time()
    # print(f'\tprep took {1000*(t_prep-t_start):.1f} ms')

    distSortedIndices = find_indices_close_to_source(source_mask, pos)#获取距离真实源最近的偶极子的索引。

    # t_prep2 = time.time()
    # print(f'\tprep2 took {1000*(t_prep2-t_prep):.1f} ms')

    sourceIndices = np.where(source_mask==1)[0]#获取活跃源的索引
    
    for n in range(n_redraw):#迭代进行重新抽样。
        #选择近源偶极子的索引。
        selectedIndicesClose = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[:closeSplit], size=numberOfActiveSources) ])
        #选择远源偶极子的索引。
        selectedIndicesFar = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[-farSplit:], size=numberOfActiveSources) ])
        #print(f'redraw {n}:\ny_true={y_true[selectedIndicesClose]}\y_est={y_est[selectedIndicesClose]}')#在循环中输出关于当前重新抽样的一些信息，以便调试或理解每次重新抽样的情况。 对于靠近源的偶极子，输出当前重新抽样的真实源向量。 对于靠近源的偶极子，输出当前重新抽样的预测源向量。
       
        fpr_close, tpr_close, _ = roc_curve(source_mask[selectedIndicesClose], y_est[selectedIndicesClose])#计算近源偶极子的ROC曲线。
   
        fpr_far, tpr_far, _  = roc_curve(source_mask[selectedIndicesFar], y_est[selectedIndicesFar])#计算远源偶极子的ROC曲线。
        
        auc_close[n] = auc(fpr_close, tpr_close)#计算近源偶极子的AUC值。
        auc_far[n] = auc(fpr_far, tpr_far)#计算远源偶极子的AUC值。
    
    auc_far = np.mean(auc_far)#计算所有重新抽样的远源AUC的平均值。
    auc_close = np.mean(auc_close)#计算所有重新抽样的近源AUC的平均值。
    #t_loops = time.time()
    # print(f'\tloops took {1000*(t_loops-t_prep2):.1f} ms')
  
    if plot_me:#如果设置了plot_me为True，则进行绘图。绘制近源和远源的ROC曲线。
        print("plotting")
        plt.figure()
        plt.plot(fpr_close, tpr_close, label='ROC_close')
        plt.plot(fpr_far, tpr_far, label='ROC_far')
        # plt.xlim(1, )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUC_close={auc_close:.2f}, AUC_far={auc_far:.2f}')
        plt.legend()
        plt.show()#显示绘制的图形。
    #print("evaluate eval_auc")
    return auc_close, auc_far#返回近源AUC和远源AUC。

def find_indices_close_to_source(source_mask, pos):
    ''' Finds the dipole indices that are closest to the active sources. 
找到距离活跃源最近的偶极子的索引，并按照它们到最近的源的距离进行排序。
这个函数首先确定活跃源的索引 (`sourceIndices`)，然后计算每个偶极子到这些活跃源的距离。最后，根据这些距离对偶极子进行排序，使得距离最近的偶极子排在前面。这个函数的输出是排序后的偶极子索引数组。
    Parameters
    -----------
    source_mask (numpy.ndarray): 一个二进制数组，指示哪些偶极子是活跃源 (1 表示活跃，0 表示非活跃)。
    simSettings : dict
        retrieved from the simulate_source function
    pos : numpy.ndarray
        list of all dipole positions in XYZ coordinates，所有偶极子位置的数组，形状为 (点数 x 维数)。

    Return
    -------
    ordered_indices : numpy.ndarray，根据每个偶极子到最近源的距离升序排序的偶极子索引数组。
        ordered list of dipoles that are near active 
        sources in ascending order with respect to their distance to the next source.
    '''

    numberOfDipoles = pos.shape[0]#获取偶极子位置数组 pos 的行数，即偶极子的数量。

    sourceIndices = np.array([i[0] for i in np.argwhere(source_mask==1)])#使用 argwhere 找到 source_mask 中值为 1 的索引，然后提取这些索引的第一个元素。创建一个包含活跃源索引的数组 sourceIndices。
    
    min_distance_to_source = np.zeros((numberOfDipoles))#创建一个长度为偶极子数量的数组 min_distance_to_source，用于存储每个偶极子到最近源的距离。
    
    
    # D = np.zeros((numberOfDipoles, len(sourceIndices)))
    # for i, idx in enumerate(sourceIndices):
    #     D[:, i] = np.sqrt(np.sum(((pos-pos[idx])**2), axis=1))
    # min_distance_to_source = np.min(D, axis=1)
    # min_distance_to_source[source_mask==1] = np.nan
    # numberOfNans = source_mask.sum()
    
    ###OLD
    numberOfNans = 0#初始化一个计数器 numberOfNans，用于跟踪在源位置处存在 NaN 值的数量。
    for i in range(numberOfDipoles):#对每个偶极子进行迭代，如果当前偶极子是活跃源，将相应的 min_distance_to_source 设置为 NaN，并递增 numberOfNans 计数器。否则，计算当前偶极子到所有活跃源的距离，将最小距离存储在 min_distance_to_source[i] 中。
        if source_mask[i] == 1:
            min_distance_to_source[i] = np.nan
            numberOfNans +=1
        elif source_mask[i] == 0:
            distances = np.sqrt(np.sum((pos[sourceIndices, :] - pos[i, :])**2, axis=1))
            min_distance_to_source[i] = np.min(distances)
        else:
            print('source mask has invalid entries')
    #print('new: ', np.nanmean(min_distance_to_source), min_distance_to_source.shape)
    ###OLD

    ordered_indices = np.argsort(min_distance_to_source)#使用 argsort 对 min_distance_to_source 进行排序，得到排序后的偶极子索引数组 ordered_indices。

    return ordered_indices[:-numberOfNans]#返回排序后的偶极子索引数组，但不包括包含 NaN 值的部分。这是因为在源位置处存在 NaN 值，它们不参与排序。

def modified_auc_metric(threshold=0.1, auc_params=dict(name='mod_auc')):
    ''' AUC metric suitable as a loss function or metric for tensorflow/keras，表明这是一个适用于 TensorFlow/Keras 的 AUC 指标，可以用作损失函数或评估指标。
    threshold（默认为 0.1）：用于对真实标签进行二值化的阈值。通过此参数，您可以灵活地定义何时将预测视为正例或负例。
    auc_params（默认为 {'name': 'mod_auc'}）：一个字典，包含要传递给 tf.keras.metrics.AUC 的其他参数。这使得用户可以定制 AUC 计算的行为。
    Parameters
    ----------
    '''
    def abs_scale(x):
        ''' Take absolute values and scale them to max=1，abs_scale 函数取输入张量的绝对值并将其缩放到最大值为 1。这是为了确保张量中的所有值都为正，并将它们缩放到相对范围 [0, 1]。
        '''
        x = K.abs(x)
        x = x / tf.expand_dims(K.max(x, axis=-1), axis=-1)
        return x

    def auc_loss(y_true, y_pred):#auc_loss 函数是一个损失函数，它计算了根据给定阈值二值化的预测与真实标签之间的AUC
        # y_true_s, y_pred_s = [y_true, y_pred]

        # Take Abs values，对真实标签和预测标签取绝对值并进行缩放。
        y_true = abs_scale(y_true)
        y_pred = abs_scale(y_pred)
        
        # Binarize Ground truth， 使用给定阈值对真实标签进行二值化。  
        y_true = y_true > threshold

        # Calc AUC，使用 tf.keras.metrics.AUC 计算二进制分类问题的AUC。
        auc = tf.keras.metrics.AUC(**auc_params)(y_true, y_pred)
        return auc
    return auc_loss#modified_auc_metric 函数返回了 auc_loss 函数，并允许指定 AUC 计算的参数（例如，设置 AUC 的名称为 'mod_auc'）

def abs_scale(x):
    ''' Take absolute values and scale them to max=1
    '''
    x = K.abs(x)
    x = x / tf.expand_dims(K.max(x, axis=-1), axis=-1)
    return x


def get_tpr_fpr(y_true, threshold_vector):#计算真正例率（True Positive Rate，TPR）和假正例率（False Positive Rate，FPR）
    #使用布尔运算计算真正例（True Positive）、真负例（True Negative）、假正例（False Positive）和假负例（False Negative）
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_true, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_true, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_true, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_true, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr#返回真正例率（True Positive Rate，TPR）和假正例率（False Positive Rate，FPR）

def auc_metric(y_true, y_pred, n_thresholds=200, epsilon=0.1):#该函数用于计算多个样本的平均 AUC 值，即 ROC 曲线下的面积。
    '''y_true：真实源向量或多个样本的真实源矩阵。y_pred：预测源向量或多个样本的预测源矩阵。n_thresholds：在 ROC 曲线中使用的阈值数量，默认为 200。
epsilon：用于对真实标签进行二值化的阈值，默认为 0.1。
    '''
    if len(y_true.shape) == 1:#检查y_true和y_pred是否为1D数组。如果是，将它们转换为2D数组
        y_true = np.expand_dims(y_true, axis=0)
        y_pred = np.expand_dims(y_pred, axis=0)
    if len(y_true.shape) > 2:#如果数组的维数大于2，将它们重新形状为2D数组
        old_dim = y_true.shape
        y_true = y_true.reshape(np.prod(y_true.shape[:-1]), y_true.shape[-1])
        y_pred = y_pred.reshape(np.prod(y_pred.shape[:-1]), y_pred.shape[-1])
    
    aucs = []#初始化一个空列表aucs，用于存储每个样本的AUC值。
    thresholds = np.linspace(0, 1, num=n_thresholds)#生成在0到1之间的n_thresholds个值的数组。
    for y_true_, y_pred_ in zip(y_true, y_pred):#遍历每个样本（y_true_和y_pred_是每个样本的向量）
        
        # Absolute and scaling，取y_true_和y_pred_的绝对值并将它们缩放到最大值为1
        y_true_ = np.abs(y_true_) / np.max(np.abs(y_true_))
        y_pred_ = np.abs(y_pred_) / np.max(np.abs(y_pred_))
        
        y_true_ = (y_true_ > epsilon).astype(int)#通过在epsilon处对y_true_进行阈值处理，将其二值化。

        fpr_vec = []
        tpr_vec = []
        for i in range(n_thresholds):#阈值循环：遍历指定数量的阈值，以计算TPR和FPR
            threshold_vector = (y_pred_ >= thresholds[i]).astype(int)
            
            tpr, fpr = get_tpr_fpr(y_true_, threshold_vector)#调用辅助函数get_tpr_fpr计算每个阈值的真正例率（TPR）和假正例率（FPR）。
            fpr_vec.append(fpr)
            tpr_vec.append(tpr)
        auc = np.abs(np.trapz(tpr_vec, x=fpr_vec))#使用梯形积分计算基于TPR和FPR值的当前样本的AUC。
        aucs.append( auc )
        #以下代码实际上是绘制ROC曲线并添加一些标签。如果取消注释，它会在每个样本的AUC计算后生成一个ROC曲线图。由于这个函数可能在训练中的多个批次中被调用，绘制多个ROC曲线图可能会导致图形过于拥挤，难以观察。因此，根据需要选择是否绘制这些图。
        #plt.figure()
        #plt.plot(fpr_vec, tpr_vec)
        #plt.ylabel('True-positive Rate')
        #plt.xlabel('False-positive Rate')
        #plt.title(f'AUC: {auc:.2f}')
        mean_auc = np.mean(aucs)
        auc_std = np.std(aucs)

    return mean_auc, auc_std
    #return np.mean(aucs)#输出返回mean_auc：所有样本的平均 AUC

def tf_get_tpr_fpr(y_true, threshold_vector):#这个函数计算给定真实标签 y_true 和阈值向量 threshold_vector 的真正例率（True Positive Rate，tpr）和假正例率（False Positive Rate，fpr）。
    true_positive = tf.cast(tf.math.equal(threshold_vector, 1) & tf.math.equal(y_true, 1), tf.int32)#计算真正例的数量。它使用 TensorFlow 的布尔运算和 tf.cast 将布尔值转换为整数。threshold_vector 中为1表示预测标签大于等于阈值，y_true 中为1表示真实标签为正例。
    true_negative = tf.cast(tf.math.equal(threshold_vector, 0) & tf.math.equal(y_true, 0), tf.int32)#计算真负例的数量。它使用 TensorFlow 的布尔运算和 tf.cast 将布尔值转换为整数。threshold_vector 中为0表示预测标签小于阈值，y_true 中为0表示真实标签为负例。
    false_positive = tf.cast(tf.math.equal(threshold_vector, 1) & tf.math.equal(y_true, 0), tf.int32)#计算假正例的数量。它使用 TensorFlow 的布尔运算和 tf.cast 将布尔值转换为整数。threshold_vector 中为1表示预测标签大于等于阈值，y_true 中为0表示真实标签为负例。
    false_negative = tf.cast(tf.math.equal(threshold_vector, 0) & tf.math.equal(y_true, 1), tf.int32)#计算假负例的数量。它使用 TensorFlow 的布尔运算和 tf.cast 将布尔值转换为整数。threshold_vector 中为0表示预测标签小于阈值，y_true 中为1表示真实标签为正例。

    tpr = K.sum(true_positive) / (K.sum(true_positive) + K.sum(false_negative))#计算真正例率（True Positive Rate，tpr），即真正例的数量除以真实正例的总数量。它使用 Keras 的 K.sum 函数来计算总和。
    fpr = K.sum(false_positive) / (K.sum(false_positive) + K.sum(true_negative))#计算假正例率（False Positive Rate，fpr），即假正例的数量除以真实负例的总数量。它使用 Keras 的 K.sum 函数来计算总和。

    return tpr, fpr#函数返回 tpr 和 fpr

def tf_auc_metric(y_true, y_pred, n_thresholds=200, epsilon=0.1):#这个函数计算给定真实标签 y_true 和预测标签 y_pred 的平均AUC（Area Under the Curve）。它通过在多个阈值上计算并平均真正例率和假正例率来实现。tf_get_tpr_fpr 函数用于计算 tpr 和 fpr。
    if len(y_true.shape) == 1:#这段代码用于确保输入的 y_true 和 y_pred 的维度正确。如果它们是一维的（例如，形状为 (n,)），则通过在第一个维度上添加一个维度来将其转换为二维（例如，形状变为 (1, n)）
        y_true = tf.expand_dims(y_true, axis=0)
        y_pred = tf.expand_dims(y_pred, axis=0)
    if len(y_true.shape) > 2:#如果输入的 y_true 和 y_pred 的维度大于二维，这段代码将它们展平为二维数组。这是因为后续的计算期望输入是二维的。
        old_dim = y_true.shape
        y_true = tf.reshape(y_true, (tf.math.reduce_prod(y_true.shape[:-1]), y_true.shape[-1]))
        y_pred = tf.reshape(y_pred, (tf.math.reduce_prod(y_pred.shape[:-1]), y_pred.shape[-1]))
    
    aucs = []
    thresholds = tf.linspace(0, 1, num=n_thresholds)#生成一个包含 n_thresholds 个值的张量，这些值在 [0, 1] 之间均匀分布，用作阈值。
    for y_true_, y_pred_ in zip(y_true, y_pred):#一个循环，对每个样本的 y_true 和 y_pred 进行操作。
        # print('go next')
        # Absolute and scaling
        y_true_ = K.abs(y_true_) / K.max(K.abs(y_true_))
        y_pred_ = K.abs(y_pred_) / K.max(K.abs(y_pred_))#这里对y_true_和y_pred_进行绝对值处理，并进行最大值归一化，确保它们的取值范围在 [0, 1]。
        
        y_true_ = tf.cast(y_true_ > epsilon, tf.int32)#将y_true_转换为整数类型，其中大于阈值 epsilon 的元素被标记为1，小于等于阈值的元素标记为0。

        fpr_vec = []
        tpr_vec = []
        for i in range(n_thresholds):#在这个循环中，对每个阈值进行操作。首先，根据当前阈值生成阈值向量。然后，调用 tf_get_tpr_fpr 函数计算真正例率（tpr）和假正例率（fpr），并将它们分别添加到 tpr_vec 和 fpr_vec 中。
            threshold_vector = tf.cast(y_pred_ >= thresholds[i], tf.int32)
            
            tpr, fpr = tf_get_tpr_fpr(y_true_, threshold_vector)
            fpr_vec.append(fpr)
            tpr_vec.append(tpr)
        auc = np.abs(np.trapz(tpr_vec, x=fpr_vec))#使用梯形法则计算 tpr-fpr 曲线下的面积（AUC），并取绝对值以确保结果为正
        print(tpr_vec)
        aucs.append( auc )#将计算得到的 AUC 添加到列表 aucs 中。
        plt.figure()
        plt.plot(fpr_vec, tpr_vec)
        plt.ylabel('True-positive Rate')
        plt.xlabel('False-positive Rate')
        plt.title(f'AUC: {auc:.2f}')
    #print("evaluate tf_auc_metric")
    return K.mean(aucs)#最终，返回AUC的平均值。

# import numpy as np
# from esinet.evaluate import eval_auc
# y_true = np.random.randn(1284)
# y_est = y_true + np.random.randn(1284)*0.5
# pos = np.random.randn(1284,3)
# eval_auc(y_true, y_est, pos)