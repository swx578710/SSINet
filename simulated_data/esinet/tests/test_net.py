#这段代码的功能是执行多个测试用例，每个测试用例都测试了"Net"神经网络模型的不同配置和参数组合。这有助于验证模型在不同条件下的性能和正确性。测试用例包括了不同的"duration_of_trial"和"model_type"参数值。通过这些测试用例，可以确保神经网络模型在不同设置下正常工作。
import pytest# 导入pytest测试框架，以便编写和运行测试用例
from .. import simulation
from .. import forward
from .. import Net#导入当前目录的上级目录中的"simulation"、"forward"和"Net"模块。这些模块包含了与神经网络模型、仿真和前向模型相关的代码。

# Crate forward model
info = forward.get_info(sfreq=100)#调用"forward"模块中的"get_info"函数，获取有关前向模型的信息，并将其存储在名为"info"的变量中。`sfreq=100`是一个参数，用于指定采样频率。
sampling = 'ico3'#选择空间模型
fwd = forward.create_forward_model(sampling=sampling)#创建前向模型，并将其存储在名为"fwd"的变量中

@pytest.mark.parametrize("duration_of_trial", [0.0, 0.1, (0.0, 0.1)])# 使用pytest的参数化装饰器，定义了名为"duration_of_trial"的测试参数，它的值可以是0.0、0.1或一个元组(0.0, 0.1)。这将在后续的测试用例中用到。
@pytest.mark.parametrize("model_type", ['lstm', 'fc', 'cnn', 'convdip'])#使用参数化装饰器，定义了名为"model_type"的测试参数，其值可以是字符串'lstm'、'fc'、'cnn'或'convdip'。
def test_net(duration_of_trial,model_type):#定义了一个名为"test_net"的测试函数，它接受两个参数，分别是"duration_of_trial"和"model_type"。这个测试函数将在后续执行多次，每次使用不同的参数值。
    settings = dict(duration_of_trial=duration_of_trial)
    sim = simulation.Simulation(fwd, info, settings=settings)#创建一个仿真对象"sim"，使用前向模型"fwd"和前向模型信息"info"，并应用上述的"settings"
    sim.simulate(n_samples=2)# 对仿真对象"sim"调用"simulate"方法，模拟生成2个样本的数据。

    # Create and train net，创建一个神经网络模型对象"net"，使用前向模型"fwd"，并指定一些模型的配置参数，包括密集层单元数、密集层层数、LSTM层层数和模型类型。
    net = Net(fwd, n_dense_units=1, n_dense_layers=1, n_lstm_layers=1, model_type=model_type)
    net.fit(sim, batch_size=1, validation_split=0.5, epochs=1)#调用"net"模型的"fit"方法，将仿真数据"sim"传递给模型进行训练。指定了批次大小、验证集划分和训练周期数。




# def test_net_temporal():#上面并不是实际预测的代码，这个代码可以用于实际预测，使用已经训练过的网络对仿真数据进行预测，并将预测结果存在y中。
    
#     # Crate forward model
#     info = forward.get_info()
#     sampling = 'ico3'
#     fwd = forward.create_forward_model(sampling=sampling)
    
#     # Simulate some little data
#     settings = dict(duration_of_trial=0.1)
#     sim = simulation.Simulation(fwd, info, settings=settings)
#     sim.simulate(n_samples=10)

#     # Create and train net
#     ann = Net(fwd, model_type="LSTM")#创建一个神经网络模型`ann`，使用前向模型`fwd`，并指定模型类型为LSTM（Long Short-Term Memory）模型。
#     ann.fit(sim, epochs=1, batch_size=1, validation_split=0.5)#使用仿真数据`sim`对神经网络模型`ann`进行训练，设置训练周期数为1，批次大小为1，以及验证集划分比例为0.5。
#     # assert ann.temporal, 'Instance of Net() must be temporal here!'
    
#     # Predict
#     y = ann.predict(sim)# 使用已训练的神经网络模型`ann`对仿真数据`sim`进行预测，获取模型的预测结果。
