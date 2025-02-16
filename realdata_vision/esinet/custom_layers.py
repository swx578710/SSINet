'''
这两个类都是 TensorFlow 中的 Layer 类，可以嵌套到其他模型中使用，用于引入注意力机制以改善模型对输入序列的处理。第一个类是通用的注意力类，而第二个类是基于 Bahdanau 注意力机制的特定实现。每个类的 `call` 方法实现了注意力的计算逻辑，包括计算注意力权重和生成上下文向量。这些注意力权重可以在模型的训练过程中动态地调整，以便模型在处理序列数据时能够更集中地关注相关部分。
'''
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, initializers, activations
# from tensorflow.keras.layers.recurrent import Recurrent, _time_distributed_dense
from tensorflow.keras.layers import RNN, TimeDistributed, InputSpec, SimpleRNN
# from tensorflow.keras.engine import 
 

class Attention(tf.keras.Model):#实现通用的注意力机制，功能：计算注意力权重，然后使用这些权重对编码器的输出进行加权求和，得到上下文向量。
    def __init__(self, units):# 输入`units` 表示注意力机制中的隐藏层单元数。
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)#`self.W1`、`self.W2` 和 `self.V` 分别是三个全连接层，用于学习注意力权重。

    def call(self, features, hidden):#`features` 是编码器的输出，`hidden` 是解码器的隐藏状态。
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
          
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights#返回上下文向量 `context_vector` 和注意力权重 `attention_weights`

class BahdanauAttention(tf.keras.layers.Layer):#计算注意力权重，然后使用这些权重对编码器的输出进行加权求和，得到上下文向量。这里采用了 Bahdanau 注意力的具体计算方式。
    def __init__(self, units, verbose=True):#`units` 表示注意力机制中的隐藏层单元数
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.verbose= verbose

    def call(self, query, values):#`query` 是解码器的隐藏状态，`values` 是编码器的所有隐藏状态序列。
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
        if self.verbose:
            print('score: (batch_size, max_length, 1) ',score.shape)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ',attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ',context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ',context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
        return context_vector, attention_weights#返回上下文向量 `context_vector` 和注意力权重 `attention_weights`。