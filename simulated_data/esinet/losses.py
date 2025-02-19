import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.spatial.distance import cdist


def combi(y_true, y_pred):#输入：`y_true` 和 `y_pred` 是真实值和预测值。输出：结合了余弦相似度损失和均方误差（MSE）的损失值。这个函数将两个损失相加。
    error_1 = tf.keras.losses.CosineSimilarity()(y_true, y_pred)
    error_2 = tf.keras.losses.MeanSquaredError() (y_true, y_pred)
    return error_1 + error_2

def reg_loss(reg=0.1):#输入：`reg` 是正则化参数。 输出：一个自定义的均方误差（MSE）损失函数，其中包含了额外的绝对值正则化。正则化项与预测值的绝对值成正比
    reg = tf.cast(reg, tf.float32)
    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred) + K.mean(K.abs(y_pred))*reg
    return mse

def nmse_loss(reg=0.05):
    ''' Weighted mean squared error (MSE) loss. A loss function that can be 
    used with tensorflow/keras which calculates the MSE with a weighting of 
    false positive predicitons. Set weight high for more conservative 
    predictions.输入：`reg` 是正则化参数.输出：一个自定义的加权均方误差（MSE）损失函数。与普通的MSE不同，该损失函数对于误差的处理更为保守，通过加权惩罚了假阳性预测。这里还包括了额外的绝对值正则化。
    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    min_val : float
        The threshold below which the target is set to zero.
    Return
    ------
    loss : loss function
    '''
    reg = tf.cast(reg, tf.float32)

    def loss(true, pred):
        # Scale to max(abs(x)) == 1
        pred = scale_mat(pred)
        true = scale_mat(true)
        
        # Calc squared error
        error = K.square(true - pred)
        

        return K.mean(error) + K.mean(K.abs(pred))*reg

    return loss

def nmae_loss(reg=0.05):
    ''' Weighted mean abs error (MSE) loss. A loss function that can be 
    used with tensorflow/keras which calculates the MSE with a weighting of 
    false positive predicitons. Set weight high for more conservative 
    predictions.输入：`reg` 是正则化参数。输出：一个自定义的加权平均绝对误差（NMAE）损失函数。
   -功能：该损失函数计算带有额外绝对值正则化项的加权平均绝对误差。该损失函数用于训练神经网络，其中通过调整权重可以对误差的不同部分进行不同程度的惩罚，以更好地满足特定问题的要求。

    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    min_val : float
        The threshold below which the target is set to zero.
    Return
    ------
    loss : loss function
    '''
    reg = tf.cast(reg, tf.float32)

    def loss(true, pred):#输入：`true` 和 `pred` 分别是真实值和预测值。输出：带有绝对值正则化项的加权平均绝对误差。功能：首先，将真实值和预测值缩放到最大绝对值为1。然后，计算绝对误差，并将其平均值与额外的绝对值正则化项相加，得到最终的损失值。
        # Scale to max(abs(x)) == 1
        pred = scale_mat(pred)
        true = scale_mat(true)
        
        # Calc squared error
        error = K.abs(true - pred)
        

        return K.mean(error) + K.mean(K.abs(pred))*reg

    return loss

def weighted_mse_loss(weight=1, min_val=1e-3, scale=True):
    ''' Weighted mean squared error (MSE) loss. A loss function that can be 
    used with tensorflow/keras which calculates the MSE with a weighting of 
    false positive predicitons. Set weight high for more conservative 
    predictions.输入：`weight` 是一个权重因子，用于惩罚误差的部分；`min_val` 是目标值低于该阈值时将其设置为零的阈值；`scale` 是一个布尔值，指示是否对输入进行缩放。 输出：一个自定义的加权均方误差（MSE）损失函数。 功能：该损失函数计算带有额外权重的加权均方误差。通过调整权重和阈值，可以对误差的不同部分进行不同程度的惩罚，以更好地满足特定问题的要求。
    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    min_val : float
        The threshold below which the target is set to zero.
    Return
    ------
    loss : loss function
    '''
    weight = tf.cast(weight, tf.float32)
    
    def loss(true, pred):#输入：`true` 和 `pred` 分别是真实值和预测值。输出：带有权重的加权均方误差。

        if scale:
            # Scale to max(abs(x)) == 1
            pred = scale_mat(pred)
            true = scale_mat(true)
        
        # Calc squared error
        error = K.square(true - pred)
        
        # False-positive weighting
        error = K.switch(K.less(K.abs(true), min_val), weight * error , error)

        return K.mean(error) 

    return loss

def weighted_huber_loss(weight=1.0, delta=1.0, min_val=1e-3, scale=True):
    ''' Weighted Huber loss. A loss function that can be 
    used with tensorflow/keras which calculates the Huber loss with 
    a weighting of false positive predicitons. Set weight high 
    for more conservative predictions.

    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    delta : float
        The delta parameter of the Huber loss. Must be non-negative.
    min_val : float
        The threshold below which the target is set to zero.

    Return
    ------
    loss : loss function
    '''

    weight = tf.cast(weight, tf.float32)
    delta = K.clip(tf.cast(delta, tf.float32), K.epsilon(), 10e2)

    def loss(true, pred):

        if scale:
            # Scale to max(abs(x)) == 1
            pred = scale_mat(pred)
            true = scale_mat(true)
        # Calc error
        differences = true-pred
        
        # Huber Loss
        error = delta * ( K.sqrt(1 + K.square(differences/delta)) -1 )

        # False-positive weighting
        error = K.switch(K.less(K.abs(true), min_val), weight * error , error)

        return K.mean(error)

    return loss

def weighted_mae_loss(w=1, min_val=1e-3, scale=True):
    ''' Weighted mean absolute error (MAE) loss. A loss function that can be 
    used with tensorflow/keras which calculates the MAE with a weighting of 
    false positive predicitons. Set weight high for more conservative 
    predictions.
    Parameters
    ----------
    weight : float
        Weighting factor which penalizes false positives.
    min_val : float
        The threshold below which the target is set to zero.
    Return
    ------
    loss : loss function
    '''
    w = tf.cast(w, tf.float32)
    
    def loss(true, pred):
        
        if scale:
            # Scale to max(abs(x)) == 1
            pred = scale_mat(pred)
            true = scale_mat(true)
        
        # MAE Loss
        error = K.abs(true - pred)
        
        # False-positive weighting
        error = K.switch(K.less(K.abs(true), min_val), w * error , error)

        return K.mean(error) 

    return loss

def scale_mat(mat):
    ''' Scale matrix such that each row has max value of 1'''
    max_vals = tf.expand_dims(K.max(K.abs(mat), axis=-1), axis=-1)
    max_vals = K.clip(max_vals, K.epsilon(), 999999999999)
    return mat / max_vals


def custom_loss(leadfield, fwd_scaler):
    def loss_batch(y_true, y_pred):
        def losss(y_true, y_pred):
            eeg_fwd = tf.matmul(leadfield, tf.expand_dims(y_pred, axis=1))
            eeg_true = tf.matmul(leadfield, tf.expand_dims(y_true, axis=1))
            eeg_fwd_scaled = eeg_fwd / K.max(K.abs(eeg_fwd))
            eeg_true_scaled = eeg_true / K.max(K.abs(eeg_true))
            
            error_mse = K.mean(K.square(y_true - y_pred))
            error_fwd = K.mean(K.square(eeg_fwd_scaled-eeg_true_scaled))
            error = error_mse + error_fwd * fwd_scaler
            return error

        batched_losses = tf.map_fn(lambda x:
                                    losss(x[0], x[1]),
                                    (y_true, y_pred),
                                    dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))  
    return loss_batch

def chamfer(pos, thresh=0.1, dtype=tf.float32):
    dist = tf.cast(cdist(pos, pos), dtype=dtype)
    pos = tf.cast(pos, dtype=dtype)
    def loss_batch(y_true, y_pred):
        def loss(y_true, y_pred):
            # print("third: ", tf.shape(y_true))
        
            # print(y_true, y_pred)
            # find indices above threshold
            idc_true = tf.where(K.abs(y_true) > K.max(K.abs(y_true)) * thresh)[:, 0]
            idc_pred = tf.where(K.abs(y_pred) > K.max(K.abs(y_pred)) * thresh)[:, 0]
            # print(idc_true, idc_pred)
            # retrieve the correct distances
            dist_true = tf.gather(dist, idc_true, axis=0)
            dist_true = tf.gather(dist_true, idc_pred, axis=1)
            
            # print(dist_true)
            
            

            lowest_dists_1 = tf.reduce_min(dist_true, axis=0)
            lowest_dists_2 = tf.reduce_min(dist_true, axis=1)

            sum_squares_1 = K.sum(K.square(lowest_dists_1))
            sum_squares_2 = K.sum(K.square(lowest_dists_2))


            error = sum_squares_1 + sum_squares_2
            # print("error on single sample and time: ", error)
            return error
        # reshaping
        new_shape = (tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.shape(y_true)[2])
        y_true = tf.reshape(y_true, new_shape)
        y_pred = tf.reshape(y_pred, new_shape)
        # print(y_true, y_pred)
        batched_losses = tf.map_fn(lambda x:
                                    loss(x[0], x[1]),
                                    (y_true, y_pred), dtype=tf.float32)
        error = K.mean(tf.stack(batched_losses))
        # print("error on all samples and all times: ", error)
        return error
    return loss_batch

def chamfer2(pos, thresh=0.2, dtype=tf.float32):
    dist = tf.cast(cdist(pos, pos), dtype=dtype)
    pos = tf.cast(pos, dtype=dtype)
    def loss_batch(y_true, y_pred):
        def loss(y_true, y_pred):
            # print("third: ", tf.shape(y_true))
        
            # print(y_true, y_pred)
            # find indices above threshold
            idc_true = tf.where(K.abs(y_true) > K.max(K.abs(y_true)) * thresh)[:, 0]
            idc_pred = tf.where(K.abs(y_pred) > K.max(K.abs(y_pred)) * thresh)[:, 0]
            # print(idc_true, idc_pred)
            # retrieve the correct distances
            dist_true = tf.gather(dist, idc_true, axis=0)
            dist_true = tf.gather(dist_true, idc_pred, axis=1)
            
            # print(dist_true)
            
            

            lowest_dists_1 = tf.reduce_min(dist_true, axis=0)
            lowest_dists_2 = tf.reduce_min(dist_true, axis=1)

            sum_squares_1 = K.mean(lowest_dists_1)
            sum_squares_2 = K.mean(lowest_dists_2)


            error = (sum_squares_1 + sum_squares_2) / 2
            # print("error on single sample and time: ", error)
            return error
        # reshaping
        # new_shape = (tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.shape(y_true)[2])
        # y_true = tf.reshape(y_true, new_shape)
        # y_pred = tf.reshape(y_pred, new_shape)
        # print(y_true, y_pred)
        batched_losses = tf.map_fn(lambda x:
                                    loss(x[0], x[1]),
                                    (y_true, y_pred), dtype=tf.float32)
        error = K.mean(tf.stack(batched_losses))
        # print("error on all samples and all times: ", error)
        return error
    return loss_batch