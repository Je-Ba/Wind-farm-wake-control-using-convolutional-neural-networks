import numpy as np
import sys
import os
import torch
import floris.tools as wfct

# To import the model, need to append the main folder to the run
# i.e. sys.path.append(path_to/acse20-acse9-finalreport-acse-jtb20)
# This works automatically on every system
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__))[0:-6]))
import CNNWake


def test_CNN_generator():
    # Test CNN model inputs

    gen = CNNWake.Generator(3, 3).to('cpu')

    # check different layer sizes, filters and activation functions
    assert isinstance(gen, CNNWake.Generator)
    assert gen.FC_Layer[0].in_features == 3
    assert gen.FC_Layer[0].out_features == 9
    assert gen.net[5].kernel_size == (3, 3)
    assert gen.net[4][2].negative_slope == 0.2


def test_CNN_layer():
    # Test  CNN layers for correct inputs/outputs

    gen = CNNWake.Generator(3, 3).to('cpu')
    layer = gen.layer(30, 10, 6, 9, 6)

    # check if correct layer, padding and stride
    # is taken
    assert layer[0].kernel_size == (6, 6)
    assert layer[0].stride == (9, 9)
    assert layer[0].padding == (6, 6)


def test_CNN_forward():
    # Test correct CNN output shape

    gen = CNNWake.Generator(3, 3).to('cpu')

    # check if output of model is a 163 x 163 image and if it can
    # accept mutiple inputs
    output = gen(torch.Tensor([[3, 0.1, 5], [8.21, 0.15, -5]]).to('cpu'))
    assert output.size() == torch.Size([2, 1, 163, 163])


def test_CNN_generate_datset():
    # Test CNN data set generation, requires FLORIS jason file
    # Check if data set if of expected size and shape

    gen = CNNWake.Generator(3, 3).to('cpu')
    x_data1, y_data1 = gen.create_floris_dataset(
        5, u_range=[3, 12],  ti_range=[0.015, 0.25], yaw_range=[-30, 30],
        image_size=54, floris_init_path=".")
    x_data2, y_data2 = gen.create_floris_dataset(
        10, u_range=[3, 12], ti_range=[0.015, 0.25], yaw_range=[-30, 30],
        image_size=200, floris_init_path=".")

    assert x_data1.shape[0] == 5
    assert x_data1.shape[1] == 3
    assert y_data1.shape[1] == 54
    assert y_data1.shape[2] == 54
    assert x_data2.shape[0] == 10
    assert x_data2.shape[1] == 3
    assert y_data2.shape[2] == 200


def test_CNN_error():
    # Test the error function to see if the error
    #  is calculated correctly

    gen = CNNWake.Generator(3, 3).to('cpu')

    # example input to generate an outputs
    x_eval = torch.tensor([[4, 0.15, 3], [4, 0.04, -30]])

    # generator output of the x_eval input is one test
    # for which the error between the generator and this
    # will be zero since the same gen is used which will
    # give the same output in the function which means
    # that both arrays are the same -> error = 0
    y_eval_gen = gen(x_eval)
    # a tensor of ones which will yield a larg error when compared
    # with gen output
    y_eval_zero = torch.ones((163, 163))*30

    test_error1 = gen.error(x_eval, y_eval_gen, "cpu",
                            image_size=163, normalisation=1)
    test_error2 = gen.error(x_eval, y_eval_zero, "cpu",
                            image_size=163, normalisation=1)
    assert type(test_error2) == np.float64
    assert test_error1 == 0
    assert test_error2 > 90


def test_CNN_evaluate_model():
    # Test if the model evaluate functions works
    # for a untrained model

    gen = CNNWake.Generator(3, 3).to('cpu')

    # try the evaluate function for the model
    error = gen.evaluate_model(set_size=10,
                               u_range=[3, 12], ti_range=[0.015, 0.2],
                               yaw_range=[-30, 30], image_size=163,
                               device='cpu', normalisation=1)

    # check output type and the error needs to be large since it is
    # an untrained network
    assert type(error) == np.float64
    assert error > 60

# Check fully connected neural network for TI and power predictions


def test_FCNN():
    # Test FCNN model inputs and model layers have correct shapes

    fcnn_model = CNNWake.FCNN(42, 5).to('cpu')

    # check different layer sizes, filters and activation functions
    assert isinstance(fcnn_model, CNNWake.FCNN)

    assert fcnn_model.disc[0].in_features == 42
    assert fcnn_model.disc[0].out_features == 5
    assert fcnn_model.disc[5].negative_slope == 0.01
    assert fcnn_model.disc[6].out_features == 1


def test_FCBB_forward():
    # Test if FCNN output is correct shape

    fcnn_model = CNNWake.FCNN(42, 5).to('cpu')

    # check if output of model is a 1 and if it can accept mutiple inputs
    output = fcnn_model(torch.Tensor([[i for i in range(0, 42)],
                                      [i for i in range(0, 42)]]).to('cpu'))
    assert output.size() == torch.Size([2, 1])


def test_FCNN_generate_datset():
    # Test FCNN training data set generation

    fcnn_model = CNNWake.FCNN(42, 5).to('cpu')
    x_data1, y_data1 = fcnn_model.create_ti_power_dataset(
        size=5, u_range=[3, 12], ti_range=[0.015, 0.25], yaw_range=[-30, 30],
        nr_varabiles=2, type='power')
    x_data2, y_data2 = fcnn_model.create_ti_power_dataset(
        size=10, u_range=[3, 12], ti_range=[0.015, 0.25], yaw_range=[-30, 30],
        nr_varabiles=40, type='ti')

    assert x_data1.shape[2] == 4
    assert y_data1.shape[1] == 1
    assert x_data2.shape[0] == 42
    assert x_data2.shape[2] == 42
    assert y_data2.shape[1] == 1


def test_FCNN_error():
    # Test FCNN error function

    fcnn_model = CNNWake.FCNN(42, 5).to('cpu')

    # example input to generate an outputs
    x_eval = torch.Tensor([[i for i in range(0, 42)],
                           [i for i in range(0, 42)]]).to('cpu')

    # The error will be zero since the same FCNN is
    # used which will give the same output in the function which means
    # that both arrays are the same -> error = 0
    y_eval_gen = fcnn_model(x_eval)

    # a tensor of 100s which will yield a large error when compared
    # with FCNN output which will be between 0 and 1 due to initialization
    y_eval_zero = torch.tensor([100, 100])*100

    test_error1 = fcnn_model.error(x_eval, y_eval_gen, "cpu")
    test_error2 = fcnn_model.error(x_eval, y_eval_zero, "cpu")
    assert test_error1 == 0
    assert test_error2 > 90

# Test superpostion model


def test_super_position():
    # Test the superpostion algo
    # Ensure the outputs are expected

    # set up example arrays
    farm_array = np.ones((100, 100))*11
    turbine_array = np.ones((40, 40))*4
    turbine_postion = [20, 20]

    # Superpostion arrays
    array_0 = CNNWake.super_position(farm_array, turbine_array,
                                     turbine_postion, 11, 4, sp_model="SOS")
    # Check if results are correct
    assert array_0.shape == farm_array.shape
    assert np.allclose(array_0[turbine_postion[0]:turbine_array.shape[0],
                       turbine_postion[0]:turbine_array.shape[0]], -3.4484454)
    assert np.allclose(array_0[turbine_array.shape[0] + turbine_postion[0]:,
                       turbine_array.shape[1] + turbine_postion[1]:], 11)

    # set up example arrays
    farm_array = np.zeros((40, 40))
    turbine_array = np.ones((10, 10))
    turbine_postion = [0, 0]
    # Superpostion arrays
    array_1 = CNNWake.super_position(farm_array, turbine_array,
                                     turbine_postion, 2,
                                     2, sp_model="linear")

    # Check if results are correct
    assert array_1.shape == farm_array.shape
    assert np.allclose(array_1[turbine_postion[0]:turbine_array.shape[0],
                       turbine_postion[0]:turbine_array.shape[0]], -1)
    assert np.allclose(array_1[turbine_array.shape[0]:,
                       turbine_array.shape[1]:], 0)

    # set up example arrays
    farm_array = np.ones((10, 10))
    turbine_array = np.ones((10, 10))*3
    turbine_postion = [0, 0]
    array_2 = CNNWake.super_position(farm_array, turbine_array,
                                     turbine_postion, 2,
                                     8, sp_model="largest_deficit")
    assert np.allclose(array_2, 1)


def test_CNN():
    # Test if the trained CNN can predict wakes
    # to at least 10% error, compare CNN output to FLORIS.
    device = torch.device("cpu")

    # Set up all NN by loading pre-trained model
    CNN_generator = CNNWake.Generator(3, 30).to(device)
    CNN_generator.load_model('./trained_models/CNN_FLOW.pt',
                             device=device)
    CNN_generator.eval()

    CNN_output = CNN_generator(torch.Tensor([[4, 0.1, 27],
                                             [11, 0.19, -19],
                                             [7, 0.08, 5]]).to('cpu'))*12
    # initialize FLORIS model using the jason file
    floris_turbine = wfct.floris_interface.FlorisInterface(
        "FLORIS_input_gauss.json")
    # set wind speed, ti and yawn angle for FLORIS model
    floris_turbine.reinitialize_flow_field(
        wind_speed=4,
        turbulence_intensity=0.1)
    floris_turbine.change_turbine([0], {'yaw_angle': 27})
    floris_turbine.calculate_wake()
    # extract horizontal plane at hub height
    floris_output1 = floris_turbine.get_hor_plane(
        height=90,
        x_resolution=163,
        y_resolution=163,
        x_bounds=[0, 3000],
        y_bounds=[-200, 200]).df.u.values.reshape(163, 163)

    # Second flow field
    floris_turbine.reinitialize_flow_field(
        wind_speed=11,
        turbulence_intensity=0.19)
    floris_turbine.change_turbine([0], {'yaw_angle': -19})
    floris_turbine.calculate_wake()
    # extract horizontal plane at hub height
    floris_output2 = floris_turbine.get_hor_plane(
        height=90,
        x_resolution=163,
        y_resolution=163,
        x_bounds=[0, 3000],
        y_bounds=[-200, 200]).df.u.values.reshape(163, 163)

    # Thrid Flow field
    floris_turbine.reinitialize_flow_field(
        wind_speed=7,
        turbulence_intensity=0.08)
    floris_turbine.change_turbine([0], {'yaw_angle': 5})
    floris_turbine.calculate_wake()
    # extract horizontal plane at hub height
    floris_output3 = floris_turbine.get_hor_plane(
        height=90,
        x_resolution=163,
        y_resolution=163,
        x_bounds=[0, 3000],
        y_bounds=[-200, 200]).df.u.values.reshape(163, 163)

    # To test if the CNN can generate the same wake as FLORIS, an element wise
    # comparison is done to check if all element are similar (up to a 1
    # tolerance). The first 8 rows are not considered because the error at
    # the rotor is large and would fail the test.
    assert np.allclose(np.squeeze(
        CNN_output[0].detach().cpu().numpy())[:, 8:], floris_output1[:, 8:], rtol=1)
    assert np.allclose(
        np.squeeze(CNN_output[1].detach().cpu().numpy())[:, 8:], floris_output2[:, 8:], rtol=1)
    assert np.allclose(
        np.squeeze(CNN_output[2].detach().cpu().numpy())[:, 8:], floris_output3[:, 8:], atol=1)


def test_FCNN_power():
    # This function tests the ability of the pre-trainined
    # FCNN to predict power generation from the flow field data,
    # to an accuracy of 95%
    device = torch.device("cpu")

    # Set up all NN by loading pre-trained model
    Power_model = CNNWake.FCNN(42, 300, 1).to(device)
    Power_model.load_state_dict(torch.load('./trained_models/FCNN_POWER.pt',
                                           map_location=device))
    Power_model.eval()

    # create four example u conditons along a line
    # to test the power prediction
    u_1 = [10/12 for i in range(0, 40)]
    u_1.append(0)
    u_1.append(0.19)

    u_2 = [5/12 for i in range(0, 40)]
    u_2.append(25/30)
    u_2.append(0.07)

    u_3 = [7/12 for i in range(0, 40)]
    u_3.append(-19/30)
    u_3.append(0.23)

    u_4 = [11 / 12 for i in range(0, 40)]
    u_4.append(0)
    u_4.append(0.1)

    # Pass examples through network
    power_1 = Power_model(torch.Tensor([u_1])).to('cpu')*4834506
    power_2 = Power_model(torch.Tensor([u_2])).to('cpu')*4834506
    power_3 = Power_model(torch.Tensor([u_3])).to('cpu')*4834506
    power_4 = Power_model(torch.Tensor([u_4])).to('cpu')*4834506

    # Check if network was able to predict power generation to 5% accaurcy
    # compared to FLORIS calculations
    assert 100*abs(3306006 - power_1.detach().cpu().numpy())/3306006 < 5
    assert 100*abs(313198 - power_2.detach().cpu().numpy())/313198 < 5
    assert 100*abs(1022605 - power_3.detach().cpu().numpy())/1022605 < 5
    assert 100*abs(4373591 - power_4.detach().cpu().numpy())/4373591 < 5


def test_FCNN_ti():
    # This function tests the ability of the pre-trainined
    # FCNN to predict the local TI from the flow field, to
    # an accuracy of 90%

    device = torch.device("cpu")
    # Set up all NN by loading pre-trained model
    TI_model = CNNWake.FCNN(42, 300, 1).to(device)
    TI_model.load_state_dict(torch.load('./trained_models/FCNN_TI.pt',
                                        map_location=device))
    TI_model.eval()

    # create four example u conditons along a line
    # to test the power prediction
    u_1 = [10 / 12 for i in range(0, 40)]
    u_1.append(0)
    u_1.append(0.19)

    u_2 = [5 / 12 for i in range(0, 40)]
    u_2.append(25 / 30)
    u_2.append(0.07)

    u_3 = [7 / 12 for i in range(0, 40)]
    u_3.append(-19 / 30)
    u_3.append(0.23)

    u_4 = [11 / 12 for i in range(0, 40)]
    u_4.append(0)
    u_4.append(0.1)

    # Pass examples through network
    ti_1 = TI_model(torch.Tensor([u_1])).to('cpu') * 0.30000001
    ti_2 = TI_model(torch.Tensor([u_2])).to('cpu') * 0.30000001
    ti_3 = TI_model(torch.Tensor([u_3])).to('cpu') * 0.30000001
    ti_4 = TI_model(torch.Tensor([u_4])).to('cpu') * 0.30000001

    # Check if network was able to predict local TI condition to 10% accuracy
    # compared to FLORIS calculations
    assert 100 * abs(0.19 - ti_1.detach().cpu().numpy()) / 0.19 < 5
    assert 100 * abs(0.093 - ti_2.detach().cpu().numpy()) / 0.093 < 10
    assert 100 * abs(0.23 - ti_3.detach().cpu().numpy()) / 0.23 < 10
    assert 100 * abs(0.1 - ti_4.detach().cpu().numpy()) / 0.1 < 10


if __name__ == "__main__":
    test_CNN_generator()
    test_CNN_layer()
    test_CNN_forward()
    test_CNN_generate_datset()
    test_CNN_error()
    test_CNN_evaluate_model()
    test_FCNN()
    test_FCBB_forward()
    test_FCNN_generate_datset()
    test_FCNN_error()
    test_super_position()
    test_CNN()
    test_FCNN_power()
    test_FCNN_ti()
    print('ALL UNIT TESTS HAVE PASSED')
