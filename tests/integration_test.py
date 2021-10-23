import numpy as np
import sys
import os
import torch
import random

# To import the model, need to append the main folder path to the run
# i.e. sys.path.append(path_to/acse20-acse9-finalreport-acse-jtb20)
# This works automatically on every system
sys.path.append(os.path.abspath(
    os.path.dirname(os.path.abspath(__file__))[0:-6]))
import CNNWake


# Test model training
def test_train_CNN():
    # Test CNN model training

    # set seeds and ensure that training is
    # less random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    devices = torch.device("cpu")
    # train small model on small training set for a few epochs
    model, loss, val_error = \
        CNNWake.train_CNN.train_CNN_model(
            nr_filters=5, nr_epochs=20, learing_rate=0.003, batch_size=100,
            train_size=200, val_size=5, image_size=163, device=devices,
            u_range=[3, 12], ti_range=[0.015, 0.25], yaw_range=[-30, 30],
            model_name='CNN.pt')

    assert isinstance(model, CNNWake.Generator)
    assert loss < 0.4
    # the validation error of less than 50 seems like a lot but the
    # training is done on the cpu and it is just test to training time
    # of less than 10 sec but error starts above 40 so should reduce
    # by a lot
    assert val_error < 40


def test_train_FCNN_power():
    # Test FCNN power model

    # set seeds and ensure that training is
    # less random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    devices = torch.device("cpu")
    # define number of u values sampled from horizontal line
    u_samples = 40
    # train small model on small training set for a few epochs
    model, loss, val_error = \
        CNNWake.train_FCNN.train_FCNN_model(
            nr_neurons=160, input_size=u_samples, nr_epochs=30,
            learing_rate=0.00005, batch_size=20, train_size=40,
            val_size=5, type='power', device=devices, u_range=[3, 12],
            ti_range=[0.015, 0.25], yaw_range=[-30, 30],
            model_name="power_model.pt")
    # check if model is correct and if the input size is correct and
    # equal to the number given to the function
    assert isinstance(model, CNNWake.FCNN)
    # Check if model accepts the correct number of inputs
    assert model.disc[0].in_features == u_samples + 2

    # loss for a untrained network is 0.5 so an test error of less than 0.1
    # shows that the model training is working
    assert loss < 0.1
    # the validation error starts of with more than 85% error so an error of
    # less than 50% for this test shows that the training is working
    assert val_error < 50


def test_train_FCNN_ti():
    # Test FCNN TI training

    # set seeds and ensure that training is
    # less random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    devices = torch.device("cpu")
    # define number of u values sampled from horizontal line
    u_samples = 20
    # train small model on small training set for a few epochs
    model, loss, val_error = \
        CNNWake.train_FCNN.train_FCNN_model(
            nr_neurons=50, input_size=u_samples, nr_epochs=40,
            learing_rate=0.0003, batch_size=5, train_size=25,
            val_size=6, type='TI', device=devices,
            u_range=[3, 12], ti_range=[0.015, 0.25], yaw_range=[-30, 30],
            model_name='./trained_models/power_model.pt')

    # check if model is correct and if the input size is correct and
    # equal to the number given to the function
    assert isinstance(model, CNNWake.FCNN)
    assert model.disc[0].in_features == u_samples + 2

    # Loss of less than 0.7 shows that the model training is working
    assert loss < 0.7
    # the validation error starts of with more than 90% error so an error of
    # less than 40% for this test shows that the training is working
    assert val_error < 40


# test power output function
def test_CNNWake_farm_power_single_turbine():
    # Test if CNNWake can predict the power of
    # a 4 example wind turbines to 5% to true power

    device = torch.device("cpu")

    # Set up all NN by loading pre-trained model
    CNN_generator = CNNWake.Generator(3, 30).to(device)
    CNN_generator.load_model('./trained_models/CNN_FLOW.pt',
                             device=device)
    CNN_generator.eval()
    Power_model = CNNWake.FCNN(42, 300, 1).to(device)
    Power_model.load_state_dict(torch.load('./trained_models/FCNN_POWER.pt',
                                           map_location=device))
    Power_model.eval()
    TI_model = CNNWake.FCNN(42, 300, 1).to(device)
    TI_model.load_state_dict(torch.load('./trained_models/FCNN_TI.pt',
                                        map_location=device))
    TI_model.eval()

    # Calculate power output of four examples using CNNWake
    power1 = CNNWake.CNNWake_farm_power([0], [100], [300], 8, 0.15,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    power2 = CNNWake.CNNWake_farm_power([0], [399], [600], 11, 0.06,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    power3 = CNNWake.CNNWake_farm_power([25], [200], [400], 4.36, 0.21,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    power4 = CNNWake.CNNWake_farm_power([-13], [200], [400], 5.87, 0.09,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    # Check if CNNWake was able to predict the power generated by every test
    # case is within 5 percent to the known true value
    assert 100*abs(1695368.64554726849 - abs(power1))/1695368.64554726849 < 5
    assert 100*abs(4373591.717498961 - abs(power2))/4373591.717498961 < 5
    assert 100*abs(187942.47740620747 - abs(power3)) / 187942.47740620747 < 5
    assert 100*abs(624533.4395056335 - abs(power4)) / 624533.4395056335 < 5


def test_CNNWake_farm_power_mutiple_turbine():
    # Test if CNNWake can predict the power of
    # a 4 example wind parks with more than 2
    # turbines to 5% to true power

    device = torch.device("cpu")

    # Set up all NN by loading pre-trained model
    CNN_generator = CNNWake.Generator(3, 30).to(device)
    CNN_generator.load_model('./trained_models/CNN_FLOW.pt',
                             device=device)
    CNN_generator.eval()
    Power_model = CNNWake.FCNN(42, 300, 1).to(device)
    Power_model.load_state_dict(torch.load('./trained_models/FCNN_POWER.pt',
                                           map_location=device))
    Power_model.eval()
    TI_model = CNNWake.FCNN(42, 300, 1).to(device)
    TI_model.load_state_dict(torch.load('./trained_models/FCNN_TI.pt',
                                        map_location=device))
    TI_model.eval()

    # Calculate the power generated by four example wind
    # parks
    power1 = CNNWake.CNNWake_farm_power([0, 0], [100, 1100],
                                        [300, 300], 6.1, 0.11,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    power2 = CNNWake.CNNWake_farm_power([-25, 25], [100, 1100],
                                        [300, 300], 6.1, 0.11,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    power3 = CNNWake.CNNWake_farm_power([-25, 15, 0], [300, 300, 850],
                                        [300, 500, 400], 9.7, 0.19,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    power4 = CNNWake.CNNWake_farm_power([0, 13, 19, 16], [50, 600, 1200, 1900],
                                        [400, 350, 450, 400],
                                        11.5, 0.09,
                                        CNN_generator, Power_model,
                                        TI_model, device)

    # Check if CNNWake power prediction is within 5% to the known value
    assert 100*abs(1178044.7762486674 - abs(power1))/1178044.7762486674 < 5
    assert 100*abs(1041185.7702935545 - abs(power2))/1041185.7702935545 < 5
    assert 100 * abs(7478873.655768376 - abs(power3)) / 7478873.655768376 < 5
    assert 100 * abs(13104825.945751127 - abs(power4)) / 13104825.945751127 < 5


def test_optimization():
    # Check if CNNWake is able optimise example wind farms

    device = torch.device("cpu")

    # Set up all NN by loading pre-trained model
    CNN_generator = CNNWake.Generator(3, 30).to(device)
    CNN_generator.load_model('./trained_models/CNN_FLOW.pt',
                             device=device)
    Power_model = CNNWake.FCNN(42, 300, 1).to(device)
    Power_model.load_state_dict(torch.load('./trained_models/FCNN_POWER.pt',
                                           map_location=device))
    Power_model.eval()
    TI_model = CNNWake.FCNN(42, 300, 1).to(device)
    TI_model.load_state_dict(torch.load('./trained_models/FCNN_TI.pt',
                                        map_location=device))
    TI_model.eval()

    # Test if CNNwake can predict optimal angle of single turbine
    yaw1, power1, timing1 = CNNWake.CNNwake_wake_steering(
        [100], [300], [-6], 7.6, 0.06, CNN_generator,
        Power_model, TI_model, device, [-30, 30], 1e-06)

    yaw1_flor, power1_flor, timing1_flor = CNNWake.FLORIS_wake_steering(
        [100], [300], [-6], 7.6, 0.06, [-30, 30], 1e-04)

    # For a single turbine, the yaw should be 0 degrees
    # A small range of 2 degrees is used to allow for tolerances
    assert 1 > yaw1[0] > -1
    assert 1 > yaw1_flor[0] > -1

    # Test if wake steering bounds work by only allowing a specific
    # range of yaw angle in the optimisation
    yaw2, power2, timing2 = CNNWake.CNNwake_wake_steering(
        [100], [300], [19], 10.4, 0.12, CNN_generator,
        Power_model, TI_model, device, [15, 25], 1e-06)

    yaw2_flor, power2_flor, timing2_flor = CNNWake.FLORIS_wake_steering(
        [100], [300], [19], 10.4, 0.12, [15, 25], 1e-04)

    assert 16 > yaw2[0] >= 15
    assert 16 > yaw2_flor[0] >= 15

    yaw3, power3, timing3 = CNNWake.CNNwake_wake_steering(
        [100], [300], [24], 5.4, 0.18, CNN_generator,
        Power_model, TI_model, device, [-14, -5], 1e-04)
    assert -5 >= yaw3[0] >= -6

    # check if it reaches 0, 0 angle for a wind park with small
    # turbine wakes which means that best yaw angle is 0, 0
    yaw4, power4, timing4 = CNNWake.CNNwake_wake_steering(
        [100, 600], [300, 300], [20, -15], 5.4, 0.21, CNN_generator,
        Power_model, TI_model, device, [-30, 30], 1e-03)
    assert -1 < yaw4[0] < 1
    assert -1 < yaw4[1] < 1

    # check if CNNwake optimisation can get same results as
    # FLORIS optimisation for a 1 x 2 wind farm
    yaw5, power5, timing5 = CNNWake.CNNwake_wake_steering(
        [100, 1100], [300, 300], [0, 0], 7.2, 0.12, CNN_generator,
        Power_model, TI_model, device, [-30, 30], 1e-05)

    yaw5_flor, power5_flor, timing5_flor = CNNWake.FLORIS_wake_steering(
        [100, 1100], [300, 300], [0, 0], 7.2, 0.12, [-30, 30], 1e-05)
    assert np.allclose(abs(yaw5), abs(yaw5_flor), atol=5)


if __name__ == '__main__':
    test_train_CNN()
    test_train_FCNN_power()
    test_train_FCNN_ti()
    test_CNNWake_farm_power_single_turbine()
    test_CNNWake_farm_power_mutiple_turbine()
    test_optimization()
    print('ALL INTEGRATION TESTS HAVE PASSED')
