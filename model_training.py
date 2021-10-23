'''
This file describes the model training process for all three models.
Firstly, import the python model CNNWake and torch.
Training the models is only one function call with the training
hyperparameter. Most of the network architecture is predefined in
the CNN_model.py and FCC_model.py file, if you want to make any changes
to the model architecture, please make there. The a single training may
take up to 60 minutes depending on your system and the hyperparameters
and training on a GPU is advised especially for the CNN network.
The training data will be generated within the training function and the
size of the training data can be specified.
The training functions show a graph of the training error after training
is finished and also save the model with the given name. The training
functions also return the trained model, training loss, and validation error
'''

import torch
import CNNWake  # import CNNwake package

# If a gpu is available use it for training, if not use cpu
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
        CNN network training
'''

# This defines size of the CNN, more filter means more parameters to train
nr_filters = 30
# Number of epochs to train the model for
nr_epochs = 20
# Optimiser learning rate
learing_rate = 0.003
# Number of datapoints used for a gradient descent step
batch_size = 100
# Size of generated training set
train_size = 60
# Size of dataset used for validation
val_size = 10
# Dimensions of the 2d array generated used the training/validation data set
# This needs to match the mode output size, (163 x 163)
image_size = 163
# Device to run the training on
device = devices
# u, ti, yaw angle range [min, max]
u_range = [3, 12]
ti_range = [0.015, 0.25]
yaw_range = [-30, 30]
# Name of the saved model after training, needs to end in .pt
model_name = 'CNN.pt'
# Number of processes to get data from RAM, more cores might speed up training
nr_workers = 0
# Path to location to the FLORIS_input_gauss.jason file, idealy just have it in
# the home directory where you run the script from
floris_path = '.'

print(f"Start training CNN")
CNN_model, CNN_loss, CNN_val_error = CNNWake.train_CNN_model(
    nr_filters=nr_filters, nr_epochs=nr_epochs, learing_rate=learing_rate,
    batch_size=batch_size, train_size=train_size, val_size=val_size,
    image_size=image_size, device=devices,
    u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
    model_name=model_name, nr_workers=nr_workers, floris_path=floris_path)
print(f"Finished CNN training, network was saved as {model_name}\n")

'''
        FCNN network training
The power and local TI predictions are made using the network architecture but
they can be trained using different training parameter.
First, the power predictor network is trained
'''
# Number of neurons in every layer, more neurons means more
# trainable parameters
nr_neurons = 60
# The input size defines how many wind speed values are extracted
# from the flow field and passed to the network
input_size = 40
nr_epochs = 40
learing_rate = 0.0009
batch_size = 30
# Number of times that the six example wind parks are used to generate data
train_size = 20
val_size = 5
# The type is either power or ti and changes the type of training data
# that is used. If set to power the dataset
type = 'power'
device = devices
u_range = [3, 12]
ti_range = [0.015, 0.25]
yaw_range = [-30, 30]
model_name = "power_model.pt"
nr_workers = nr_workers
floris_path = '.'

print(f"Start training FCNN power")
power_model, power_loss, power_val_error = CNNWake.train_FCNN_model(
    nr_neurons=nr_neurons, input_size=input_size, nr_epochs=nr_epochs,
    learing_rate=learing_rate, batch_size=batch_size, train_size=train_size,
    val_size=val_size, type=type, device=devices, u_range=u_range,
    ti_range=ti_range, yaw_range=yaw_range, model_name=model_name,
    nr_workers=nr_workers, floris_path=floris_path)
print(f"Finished FCNN power training, network was saved as {model_name}\n")

'''
Lastly, training the network for the local TI predictions
'''

nr_neurons = 60
input_size = 40
nr_epochs = 40
learing_rate = 0.0009
batch_size = 30
train_size = 20
val_size = 5
# The type must be changed to ti if the local ti should be used in the
# data set generation
type = 'ti'
device = devices
u_range = [3, 12]
ti_range = [0.015, 0.25]
yaw_range = [-30, 30]
model_name = "ti_model.pt"
nr_workers = nr_workers
floris_path = '.'

print(f"Start training FCNN TI")
ti_model, ti_loss, ti_val_error = CNNWake.train_FCNN_model(
    nr_neurons=nr_neurons, input_size=input_size, nr_epochs=nr_epochs,
    learing_rate=learing_rate, batch_size=batch_size, train_size=train_size,
    val_size=val_size, type=type, device=devices, u_range=u_range,
    ti_range=ti_range, yaw_range=yaw_range, model_name=model_name,
    nr_workers=nr_workers, floris_path=floris_path)
print(f"Finished FCNN TI training, network was saved as {model_name}\n")
