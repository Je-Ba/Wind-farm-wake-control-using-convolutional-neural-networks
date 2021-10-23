import torch
import floris.tools as wfct
import CNNWake  # import CNNwake package


'''              CNNwake yaw angle optimisation
To perform a yaw angle optimisation using CNNwake 4 steps are needed:
1. Ensure that you have pip installed all required external packages
2. Select to run Neural networks on either CPU or GPU
3. Load CNN, power and TI networks from the .pf files and set it to
   evaluation mode
4. Call CNNWake.CNNwake_wake_steering with the desired wind park
   layout, inflow condition, Networks, required yaw bounds and tolerance

To compare the optimized yaw angle with a solution generated using FLORIS
please call CNNWake.FLORIS_wake_steering with desired wind park layout,
inflow condition, yaw angle bounds, tolerance and path to FLORIS.jason file

An example for both functions is given below
'''

# Select to run the model on either CPU ot GPU, if models should be run
# on GPU please replace cpu with cude below
device = torch.device("cpu")

# initialise network to generate turbine wake
nr_input_parameters = 3  # three input to the model: u, ti and yaw
filters = 30  # filters used in deconvolutional layers
CNN_generator = CNNWake.Generator(nr_input_parameters, filters).to(device)
# Load trained model and set it to evaluation mode
CNN_generator.load_model('./trained_models/CNN_FLOW.pt', device=device)
CNN_generator.eval()

# initialise network to predict power
nr_input_values = 42  # Number of input values
nr_neurons = 300  # Number of neurons in every layer
nr_output = 1  # Number of outputs from model
Power_model = CNNWake.FCNN(nr_input_values, nr_neurons, nr_output).to(device)
# Load trained model and set it to evaluation mode
Power_model.load_model('./trained_models/FCNN_POWER.pt', device=device)
Power_model.eval()

# initialise network to local turbulent intensities
nr_input_values = 42  # Number of input values
nr_neurons = 300  # Number of neurons in every layer
nr_output = 1  # Number of outputs from model
TI_model = CNNWake.FCNN(nr_input_values, nr_neurons, nr_output).to(device)
# Load trained model and set it to evaluation mode
TI_model.load_model('./trained_models/FCNN_TI.pt', device=device)
TI_model.eval()

'''
Call CNNwake_wake_steering with all the required arguments:

List of all x and y locations of all turbines in meter. Please ensure that
   the x location list is sorted from upstream to downstream.
   All y postions need to be more than 300 meter above 0, this is to ensure
   that no turbine wake is cut of by the model
List of initial yaw angle for every turbine, set all to 0 if no
   approximate solution is known.
Wind farm inlet wind speed [m/s]
Turbulent intensity at wind farm inlet [%/100]
Neural Networks loaded above in the same order
device defined above which determines where to run the model on
Yaw angle bounds, from smallest to largest, please ensure that
   model was trained with the corresponding angles
Tolerance for optimization

The function will return the optimal yaw angle as a list, the
energy produced at this setting and time taken for optimization.
'''
print("         CNNwake yaw angle optimisation")
yaw, power, timing = CNNWake.CNNwake_wake_steering(
        [100, 100, 700, 700],
        [300, 800, 300, 800],
        [0, 0, 0, 0],
        11.6, 0.06, CNN_generator, Power_model, TI_model,
        device, [-30, 30], 1e-07)

print(f'CNNwake optimized yaw angle are: {yaw}')
print(f'CNNwake optimized power output: {power}\n')

'''
To validate the solution, CNNWake.FLORIS_wake_steering can be used.
This function works similarly than but uses FLORIS instead instead of
CNNwake. The inputs to the function are:

list of all x and y locations of all turbines in meter. Please ensure that
   the x location list is sorted from upstream to downstream.
   All y postions need to be more than 300 meter above 0, this is to ensure
   that no turbine wake is cut of by the model
List of inital yaw angle for every turbine, set all to 0 if no
   approximate solution is known.
Wind farm inlet wind speed [m/s]
Turbulent intensity at wind farm inlet [%/100]
Yaw angle bounds, from smallest to largest, please ensure that
   model was trained with the corresponding angles
Tolerance of optimization
Path to location for FLORIS.jason file

The function will return the optimal yaw angle as a list, the
#energy produced at this setting and time taken for optimization.
'''

print("         FLROIS yaw angle optimisation")

yaw_floris, power_floris, timing_floris = CNNWake.FLORIS_wake_steering(
        [100, 100, 700, 700],
        [300, 800, 300, 800],
        [0, 0, 0, 0],
        11.6, 0.06, [-30, 30], 1e-07, './')

print(f'FLORIS optimized yaw angle are: {yaw_floris}')
print(f'FLORIS optimized power output: {power_floris}\n')


'''              Compare CNNwake and FLORIS solution
To compare CNNwake and FLORIS solution 4 steps are needed:
1. Ensure that you have pip installed all required external packages
2. Select to run Neural networks on either CPU or GPU
3. Load CNN, power and TI networks from the .pf files and set it to
   evaluation mode
4. Call CNNWake.Compare_CNN_FLORIS with the desired wind park
   layout, inflow condition, Networks, required yaw bounds and tolerance

The function will generate two plots, one plot of the flow fields/pixel
percentageplot and one plot comparing the power and local TI predictions
for every turbine. It will also print the APWP error and power generation
and local TI error.
'''

print("         Compare FLORIS and CNNwake")
yaw, power = CNNWake.Compare_CNN_FLORIS(
        [100, 100, 700, 700],
        [300, 800, 300, 800],
        [20, -20, -20, 20],
        7.6, 0.12, CNN_generator, Power_model, TI_model,
        device, florisjason_path='', plot=True)
print("\n")


'''
                CNNwake wind farm power
Other functions that might be of interest are FLORIS_farm_power and
CNNWake_farm_power which both calculate the power generated by a wind
park. This is the function that is called by the optimiser to find
the best yaw angle. The returned power is negative due to the
requirement to find the minimum so just  get the absolute value of
the returned power.
The main difference when calling this function is that the list of
yaw angle needs to be the first argument followed by the arguments
just as seen above.

'''

print('          CNNwake wind park power calculation')
CNNwake_power = CNNWake.CNNWake_farm_power(
        [12, -28, 6, 5],
        [200, 200, 700, 700],
        [400, 800, 400, 800],
        7.6, 0.12, CNN_generator, Power_model, TI_model, device)
print(f'Wind park power prediction generated by CNNwake: {abs(CNNwake_power)}')

# To use the FLORIS function that find the power produced by the
# wind park, the FLORIS interface needs to be passed to the
# function

floris_park = wfct.floris_interface.FlorisInterface(
        "FLORIS_input_gauss.json")
florsi_power = CNNWake.FLORIS_farm_power(
        [12, -28, 6, 5],
        [200, 200, 700, 700],
        [400, 800, 400, 800],
        7.6, 0.12, floris_park)

print(f'Wind park power prediction generated by FLORIS: {abs(florsi_power)}')
