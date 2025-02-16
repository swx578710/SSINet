import pytest#导入Python的测试框架pytest。
from .. import simulation
from .. import forward

fwd = forward.create_forward_model(sampling='ico3')#创建一个名为`fwd`的脑电前向模型，采用'ico3'采样策略。
info = forward.get_info()

def test_create_fwd_model():#定义一个测试函数`test_create_fwd_model`，用于测试前向模型的创建。
    sampling = 'ico1'
    fwd = forward.create_forward_model(sampling=sampling)

def test_create_info():#定义一个测试函数`test_create_info`，用于测试获取前向模型信息。
    info = forward.get_info()#获取前向模型的信息。

#使用pytest的参数化装饰器，为接下来的测试函数定义不同的参数组合，以进行多次测试。这些参数包括仿真数据的属性，如源数量、持续时间、采样频率等。
@pytest.mark.parametrize("number_of_sources", [2,])#源的个数
@pytest.mark.parametrize("extents", [(1,20),])#源的直径
@pytest.mark.parametrize("amplitudes", [1,])#源的振幅
@pytest.mark.parametrize("shapes", ['mixed', 'gaussian', 'flat'])
@pytest.mark.parametrize("duration_of_trial", [0, 0.1, (0, 0.1)])#源的持续时间
@pytest.mark.parametrize("sample_frequency", [100,])
@pytest.mark.parametrize("target_snr", [5, ])
@pytest.mark.parametrize("beta", [1, ])
@pytest.mark.parametrize("method", ['standard', 'noise', 'mixed'])
@pytest.mark.parametrize("source_spread", ['region_growing', 'spherical', 'mixed'])
@pytest.mark.parametrize("source_number_weighting", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_simulation( number_of_sources, extents, amplitudes,
        shapes, duration_of_trial, sample_frequency, target_snr, beta,
        method, source_spread, source_number_weighting, parallel):

    settings = {
            'number_of_sources': number_of_sources,
            'extents': extents,
            'amplitudes': amplitudes,
            'shapes': shapes,
            'duration_of_trial': duration_of_trial,
            'sample_frequency': sample_frequency,
            'target_snr': target_snr,
            'beta': beta,
            'method': method,
            'source_spread': source_spread,
            'source_number_weighting': source_number_weighting
        }

    sim = simulation.Simulation(fwd, info, settings=settings, parallel=parallel)#创建一个仿真对象`sim`，使用前向模型`fwd`、前向模型信息`info`以及指定的仿真数据设置。
    sim.simulate(n_samples=2)# 对仿真对象`sim`调用`simulate`方法，生成2个样本的仿真数据。


def test_simulation_add():#定义一个测试函数`test_simulation_add`，用于测试仿真数据对象的加法操作。
    sim_a = simulation.Simulation(fwd, info).simulate(n_samples=2)
    sim_b = simulation.Simulation(fwd, info).simulate(n_samples=2)
    sim_c = sim_a + sim_b#执行仿真数据对象`sim_a`和`sim_b`的加法操作，将它们合并成一个新的仿真数据对象`sim_c`。
    

def create_forward_model_test(pth_fwd='temp/ico2/', sampling='ico2'):#定义一个名为`create_forward_model_test`的测试函数，用于测试创建前向模型的功能。
    # Create a forward model
    forward.create_forward_model(pth_fwd, sampling=sampling)
    info = forward.get_info(sfreq=100)#获取前向模型的信息，设置采样频率为100Hz。
    fwd = forward.create_forward_model(info=info)
    fwd_free = forward.create_forward_model(info=info, fixed_ori=False)#创建另一个前向模型`fwd_free`，禁用方向定向（fixed_ori设置为False）。
        
 
    
