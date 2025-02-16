import numpy as np
from copy import deepcopy
#代码实现了eLORETA（exact Low-Resolution Electromagnetic Tomography）逆问题求解的算法，用于估计脑源活动。
#这组代码主要用于基于脑电数据计算eLORETA逆问题求解。在eLORETA中，通过优化权重矩阵 `D`，该算法试图找到最接近实际源活动的解。这可以用于定位脑中活动的空间位置。
def centeringMatrix(n):
    ''' Centering matrix, which when multiplied with a vector subtract the mean of the vector.**`centeringMatrix` 函数**:
   - 输入：整数 `n`，表示矩阵的维度。
   - 输出：中心化矩阵，当与一个向量相乘时，会从向量中减去其均值。
'''
    C = np.identity(n) - (1/n) * np.ones((n, n))
    return C

def eloreta(x, leadfield, tikhonov=0.05, stopCrit=0.005, verbose=True):
    ''' 输入：
     - `x`：观测到的电位数据。
     - `leadfield`：感知器（电极）到源之间的前向矩阵。
     - `tikhonov`：Tikhonov正则化参数。
     - `stopCrit`：迭代停止的阈值。
     - `verbose`：是否输出详细的调试信息。
   - 输出：通过eLORETA算法计算得到的源活动估计。
    '''
    D, C = calc_eloreta_D(leadfield, tikhonov, stopCrit=stopCrit, verbose=verbose)
    
    K_elor = np.matmul( np.matmul(np.linalg.inv(D), leadfield.T), np.linalg.inv( np.matmul( np.matmul( leadfield, np.linalg.inv(D) ), leadfield.T) + (tikhonov**2 * C) ) )

    y_est = np.matmul(K_elor, x)
    return y_est

def calc_eloreta_D(leadfield, tikhonov, stopCrit=0.005, verbose=True):
    ''' Algorithm that optimizes weight matrix D as described in 
        Assessing interactions in the brain with exactlow-resolution electromagnetic tomography; Pascual-Marqui et al. 2011 and
        https://www.sciencedirect.com/science/article/pii/S1053811920309150- 
        输入：
     - `leadfield`：感知器到源的前向矩阵。
     - `tikhonov`：Tikhonov正则化参数。
     - `stopCrit`：迭代停止的阈值。
     - `verbose`：是否输出详细的调试信息。
   - 输出：eLORETA算法中的权重矩阵 `D` 和调整矩阵 `C`。
        '''
    numberOfElectrodes, numberOfVoxels = leadfield.shape
    # initialize weight matrix D with identity and some empirical shift (weights are usually quite smaller than 1)
    D = np.identity(numberOfVoxels)
    H = centeringMatrix(numberOfElectrodes)
    if verbose:
        print('Optimizing eLORETA weight matrix W...')
    cnt = 0
    while True:
        old_D = deepcopy(D)
        if verbose:
            print(f'\trep {cnt+1}')
        C = np.linalg.pinv( np.matmul( np.matmul(leadfield, np.linalg.inv(D)), leadfield.T ) + (tikhonov * H) )
        for v in range(numberOfVoxels):
            leadfield_v = np.expand_dims(leadfield[:, v], axis=1)
            D[v, v] = np.sqrt( np.matmul(np.matmul(leadfield_v.T, C), leadfield_v) )
        
        averagePercentChange = np.abs(1 - np.mean(np.divide(np.diagonal(D), np.diagonal(old_D))))
        if verbose:
            print(f'averagePercentChange={100*averagePercentChange:.2f} %')
        if averagePercentChange < stopCrit:
            if verbose:
                print('\t...converged...')
            break
        cnt += 1
    if verbose:
        print('\t...done!')
    return D, C

def mne_eloreta(sim):
    ''' Calculates the inverse solution based on the sim object.
    '''
    eeg, sources = Net._handle_data_input((sim,))
    print(eeg, sources)
    print(eeg[0].get_data().shape)

    pass
