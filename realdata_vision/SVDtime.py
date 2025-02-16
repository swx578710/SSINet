import numpy as np
import random
from matplotlib import pyplot as plt


def perform_svd(data):
    # Transpose data so that rows are timesteps and columns are channels
    U, s, Vt = np.linalg.svd(data.T, full_matrices=False)
    return U, s, Vt

def select_top_k_components(s, threshold=0.8):
    total_variance = np.sum(s)
    cumulative_variance = np.cumsum(s)
    k = np.argmax(cumulative_variance >= total_variance * threshold) + 1  # +1 因为索引从0开始
    return k

def generate_time_series(U, s, k, signal_length):

    # 随机选择n值
    n = random.randint(1, k)

    # 随机选择 n 个基函数的索引
    selected_indices = np.random.choice(k, n, replace=False)

    # 随机选择每个选中基函数的权重
    weights = np.random.rand(n)

    # 使用选中的基函数和权重生成时间序列
    weighted_components = U[:, selected_indices] @ (weights[:, None] * s[selected_indices, None].T)

    # 将加权的基函数相加得到最终的时间序列
    new_signal = np.sum(weighted_components, axis=1)

    # # 随机选择一个起始列索引
    # start_col = np.random.randint(0, new_signal.shape[1] - signal_length + 1)
    # # 裁剪出子矩阵
    # submatrix = new_signal[start_col:start_col + signal_length]

    submatrix = new_signal

    return submatrix


def custom_signal_generator(signal_length):
    # Step 1: Perform SVD on the transposed data matrix

    A = np.load('x_oddball.npy') # 数据的维度是（时间*通道）

    U, s, Vt = perform_svd(A)

    # Step 2: Select the number of top components to use
    k = select_top_k_components(s)

    # Step 3: Generate new time series based on the top k components
    new_signal = generate_time_series(U, s, k, signal_length)

    return new_signal

# 假设 A 是某个二维时间序列，行为通道，列为时间点
# A = np.random.randn(10, 1000)  # 10个通道，每个通道1000个时间点

if __name__ == "__main__":
    # 生成长度为 500 的时间序列
    new_time_series = custom_signal_generator(600)
    print(new_time_series)
    print(new_time_series.shape)

    plt.plot(new_time_series)
    plt.show()
