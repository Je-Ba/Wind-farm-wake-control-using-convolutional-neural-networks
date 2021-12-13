import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import floris.tools as wfct
from superposition import super_position
from optimisation import FLORIS_wake_steering, CNNwake_wake_steering
from superposition import FLORIS_farm_power, CNNWake_farm_power


__author__ = "Jens Bauer"
__copyright__ = "Copyright 2021, CNNwake"
__credits__ = ["Jens Bauer"]
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jens.bauer20@imperial.ac.uk"
__status__ = "Development"



def visualize_turbine(plane, domain_size, nr_points, title="", ax=None):
    """
    Function to plot the flow field around a single turbine

    Args:
        plane (2d numpy array): Flow field around turbine
        domain_size (list or numpy array): x and y limits of the domain,
            the first two values correspond to min and max of x and
            similar for the y values [x_min, x_max, y_min, y_max]
        nr_points (list or numpy array): Nr. of points in the array
        title (str, optional): Title of the graph. Defaults to "".
        ax (ax.pcolormesh, optional): Pyplot subplot class,
            adds the plot to this location.

    Returns:
        ax.pcolormesh: Image of the flow field
    """
    # create mesh grid for plotting
    x = np.linspace(domain_size[0], domain_size[1], nr_points[0])
    y = np.linspace(domain_size[2], domain_size[3], nr_points[1])
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Plot the cut-through
    im = ax.pcolormesh(x_mesh, y_mesh, plane, shading='auto', cmap="coolwarm")
    ax.set_title(title)
    # Make equal axis
    ax.set_aspect("equal")

    return im


def visualize_farm(
        plane, nr_points, size_x, size_y, title="", ax=None, vmax=False):

    """
    Function to plot flow-field around a wind farm.

    Args:
        plane (2d numpy array): Flow field of wind farm
        nr_points (list or np array): List of nr of points in x and y
        size_x (int): Size of domain in x direction (km)
        size_y (int): Size of domain in y direction (km)
        title (str, optional): Title of the plot. Defaults to "".
        ax (ax.pcolormesh, optional): Pyplot subplot class, adds the plot
            to this location.
        vmax (bool, optional): Maximum value to plot. If false,
            the max value of the plane is used a vmax

    Returns:
        ax.pcolormesh: Image of the flow field around the wind farm
    """
    x = np.linspace(0, size_x, nr_points[0])  # this is correct!
    y = np.linspace(0, size_y, nr_points[1])
    x_mesh, y_mesh = np.meshgrid(x, y)

    # if no vmax is set, use the maximum of plane
    if vmax is False:
        vmax = np.max(plane)

    # Plot the cut-through
    im = ax.pcolormesh(x_mesh, y_mesh, plane,
                       shading='auto', cmap="coolwarm", vmax=vmax)
    ax.set_title(title)
    # Make equal axis
    ax.set_aspect("equal")

    return im


def Compare_CNN_FLORIS(
        x_position, y_position, yawn_angles, wind_velocity, turbulent_int,
        CNN_generator, Power_model, TI_model, device,
        florisjason_path='', plot=False):
    """
    Generates the wind field around a wind park using the neural networks.
    The individual wakes of the turbines are calculated using thee CNN and
    superimposed onto the wind farm flow field using a super-position model.
    The energy produced by the turbines are calcuated using another fully
    connected network from the flow data just upstream the turbine.
    The functions generates the same wind park flow field using FLORIS so that
    the two solutions can be compared when plot = True is set.

    Args:
        x_position (list): 1d array of x locations of the wind turbines in m.
        y_position (list): 1d array of  y locations of the wind turbines in m.
        yawn_angles (list): 1d array of yaw angles of every wind turbine.
        wind_velocity (float): Free stream wind velocity in m/s.
        turbulent_int (float): Turbulent intensity in percent.
        device (torch.device): Device to store and run the neural network on,
            cpu or cuda
        florisjason_path (string): Location of the FLORIS jason file
        plot (bool, optional): If True, the FLORIS and CNN solution will
            be plotted and compared.

    Returns:
        numpy array: Final 2d array of flow field around the wind park.
    """

    # Define the x and y length of a single cell in the array
    # This is set by the standard value used in FLORIS wakes
    dx = 18.4049079755
    dy = 2.45398773006
    # Set the maximum length of the array to be 3000m and 400m
    # more than the maximum x and y position of the wind park
    # If a larger physical domain was used change adapt the values
    x_max = np.max(x_position) + 3000
    y_max = np.max(y_position) + 300
    # Number of cells in x and y needed to create a 2d array of
    # that is x_max x y_max using dx, dy values
    Nx = int(x_max / dx)
    Ny = int(y_max / dy)
    # Initialise a 2d array of the wind park with the
    # inlet wind speed
    farm_array = np.ones((Ny, Nx)) * wind_velocity

    # set up FLORIS model
    floris_model = wfct.floris_interface.FlorisInterface(
        florisjason_path + "FLORIS_input_gauss.json")
    floris_model.reinitialize_flow_field(
        layout_array=[x_position, np.array(y_position)])
    for _ in range(0, len(x_position)):
        floris_model.change_turbine([_], {'yaw_angle': yawn_angles[_],
                                          "blade_pitch": 0.0})
    floris_model.reinitialize_flow_field(wind_speed=wind_velocity,
                                         turbulence_intensity=turbulent_int)
    start_t = time.time()
    # Calcuate using FLORIS and extract 2d flow field
    floris_model.calculate_wake()
    print(f"Time taken for FLORIS to generate"
          f" wind park: {time.time() - start_t:.3f}")
    floris_plane = floris_model.get_hor_plane(
        height=90, x_resolution=Nx, y_resolution=Ny, x_bounds=[0, x_max],
        y_bounds=[0, y_max]).df.u.values.reshape(Ny, Nx)
    floris_power = floris_model.get_turbine_power()
    floris_ti = floris_model.get_turbine_ti()
    # print(floris_power, floris_ti)

    power_CNN = []
    ti_CNN = []
    t = time.time()
    with torch.no_grad():
        # Do CNNwake cautions
        for i in range(len(x_position)):
            # determine the x and y cells that the turbine center is at
            turbine_cell = [int((x_position[i]) / dx),
                            int((y_position[i] - 200) / dy)]

            t1 = time.time()
            # extract wind speeds along the rotor, 60 meters upstream
            u_upstream_hub = farm_array[
                             turbine_cell[1] + 45: turbine_cell[1] + 110,
                             turbine_cell[0] - 3]
            # Do an running average, this is done because CNNwake has slight
            # variations in the u predictions, also normalise the u values
            u_power = [
                ((u_upstream_hub[i - 1] + u_upstream_hub[i] +
                  u_upstream_hub[i + 1]) / 3) / 12 for
                i in np.linspace(5, 55, 40, dtype=int)]

            u_power = np.append(u_power, yawn_angles[i] / 30)
            u_power = np.append(u_power, turbulent_int)

            # The local TI does not change from inlet TI if the turbine
            # is not covered by a wake, therefore check if if all values
            # in u_list_hub are the same -> means no wake coverage
            # Local TI also depends on yaw, if yaw is less than 12° and
            # turbine is not in wake -> use inlet TI for local TI
            if np.allclose(u_power[0], u_power[0:-3],
                           rtol=1e-02, atol=1e-02) and abs(u_power[-2]) < 0.4:
                # print("Turbine in free stream, set ti to normal")
                ti = turbulent_int
            else:
                ti = TI_model((torch.tensor(u_power).float().to(device))).detach().cpu().numpy() * 0.30000001192092896
                # regulate TI to ensure it is not to different from free stream
                if ti < turbulent_int * 0.7:
                    # print(f"TI REGULATED 1 AT {i}")
                    ti = turbulent_int * 1.5
                # clip ti values to max and min trained
                ti = np.clip(ti, 0.015, 0.25).item(0)
            ti_CNN.append(ti)

            u_power[-1] = ti
            energy = Power_model(torch.tensor(u_power).float().to(device)).detach().cpu().numpy() * 4834506
            power_CNN.append(energy[0])

            hub_speed = np.round(np.mean(u_upstream_hub), 2)
            turbine_condition = [[hub_speed, ti, yawn_angles[i]]]

            turbine_field = CNN_generator(torch.tensor(turbine_condition).float().to(device))

            # Use CNN to calculate wake of individual trubine
            # Since CNN output is normalised,
            # mutiply by 12 and create a numpy array
            turbine_field = turbine_field[0][0].detach().cpu().numpy() * 12
            # Place wake of indivual turbine in the farm_array
            farm_array = super_position(
                farm_array, turbine_field, turbine_cell, hub_speed,
                wind_velocity, sp_model="SOS")

    # print information
    print(f"Time taken for CNNwake to generate wind park: {time.time() - t:.3f}")

    print(f"CNNwake power prediction error: "
          f"{100 * np.mean(abs(np.array(floris_power) - np.array(power_CNN)) / np.array(floris_power)):.2f} %")

    print(f"CNNwake TI prediction error: {100 * np.mean(abs(np.array(floris_ti) - np.array(ti_CNN)) / np.array(floris_ti)):.2f} %")

    print(f"APWP error: {100 * np.mean(abs(floris_plane - farm_array) / np.max(floris_plane)):.2f}")

    if plot:
        plt.rcParams.update({'font.size': 16})
        # Plot wake fields of both wind farms and error field
        fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(20, 49))
        im1 = visualize_farm(farm_array, nr_points=[Nx, Ny], size_x=x_max,
                             size_y=y_max, title="CNNwake", ax=axarr[0])
        im2 = visualize_farm(floris_plane, nr_points=[Nx, Ny], size_x=x_max,
                             size_y=y_max, title="FLORIS", ax=axarr[1])
        im3 = visualize_farm(
            (100 * abs(floris_plane - farm_array) / np.max(floris_plane)),
            nr_points=[Nx, Ny], size_x=x_max, size_y=y_max,
            title="Pixel wise percentage error ", ax=axarr[2], vmax=20)

        col1 = fig.colorbar(im1, ax=axarr[0])
        col1.set_label('m/s', labelpad=15, y=1.06, rotation=0)
        col2 = fig.colorbar(im2, ax=axarr[1])
        col2.set_label('m/s', labelpad=15, y=1.06, rotation=0)
        col3 = fig.colorbar(im3, ax=axarr[2])
        col3.set_label('%', labelpad=11, y=0.9, rotation=0)

        axarr[2].set_xlabel('m', fontsize=15)
        axarr[0].set_ylabel('m', labelpad=9, rotation=0, y=0.4, fontsize=15)
        axarr[1].set_ylabel('m', labelpad=9, rotation=0, y=0.4, fontsize=15)
        axarr[2].set_ylabel('m', labelpad=9, rotation=0, y=0.4, fontsize=15)

        # Plot TI and Power of every turbine for FLORIS adn CNNNwake
        fig, axarr = plt.subplots(2, figsize=(9, 9))
        axarr[0].plot(range(1, len(x_position) + 1),
                      np.array(power_CNN)/1.e06, 'o--', label="CNNwake")
        axarr[0].plot(range(1, len(x_position) + 1),
                      np.array(floris_power)/1.e06, 'o--', label="FLORIS")

        axarr[1].plot(range(1, len(x_position) + 1),
                      np.array(ti_CNN), 'o--', label="CNNwake")
        axarr[1].plot(range(1, len(x_position) + 1),
                      floris_ti, 'o--', label="FLORIS")

        axarr[0].set_ylabel('Power output [MW]', fontsize=15)

        axarr[1].set_ylabel('Local TI [%]', fontsize=15)
        axarr[1].set_xlabel('Turbine Nr.', rotation=0, fontsize=15)

        axarr[1].legend()
        axarr[0].legend()

        plt.show()

    return farm_array, floris_plane


if __name__ == '__main__':
    # To run individual CNNWake files, the imports are not allowed to be
    # relative. Instead of:  from .superposition import super_position
    # it needs to be:  from superposition import super_position, for all CNNWake imports
    # also import all NNs
    from CNN_model import Generator
    from FCC_model import FCNN
    from superposition import super_position

    # Set up/load all NNs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    CNN_generator = Generator(3, 30).to(device)
    CNN_generator.load_model('./trained_models/CNN_FLOW.pt', device=device)
    CNN_generator = CNN_generator.to()
    CNN_generator.eval()
    # the first forward pass is super slow so do it outside loop and use the
    # output for a simple assert test
    example_out = CNN_generator(torch.tensor([[4, 0.1, 20]]).float().to(device))
    assert example_out.size() == torch.Size([1, 1, 163, 163])

    Power_model = FCNN(42, 300, 1).to(device)
    Power_model.load_state_dict(torch.load('./trained_models/FCNN_POWER.pt', map_location=device))
    Power_model.eval()
    # the first forward pass is super slow so do it outside loop and use the
    # output for a simple assert test
    energy = Power_model(torch.tensor([i for i in range(0, 42)]).float().to(device))
    assert energy.size() == torch.Size([1])

    TI_model = FCNN(42, 300, 1).to(device)
    TI_model.load_state_dict(torch.load('./trained_models/FCNN_TI.pt', map_location=device))
    TI_model.eval()
    # the first forward pass is super slow so do it outside loop and use the
    # output for a simple assert test
    TI = TI_model(torch.tensor([i for i in range(0, 42)]).float().to(device))
    assert TI.size() == torch.Size([1])

    # Compare a single wind farm, this will show the wake, energy and local TI
    # for every turbine and compare it to FLORIS
    '''farm, a = Compare_CNN_FLORIS([100, 100, 700, 700, 1200, 1200],
                                 [300, 800, 1300, 550, 1050, 300],
                                 [0, 0, 0, 0, 0, 0, 0], 11.6, 0.06,
                                 CNN_generator, Power_model,
                                 TI_model, device, plot=True)'''


    def heatmap(xs, ys, res=10):
        """
        Assess the performance of the DNN vs FLORIS on
        parametric optimiser calls for a wide range of
        inlet speed and turbulence intensity for a
        specific array configuration.
        Args:
            xs (numpy array of floats) Turbine x coordinates.
            ys (numpy array of floats) Turbine y coordinates.
            res (int, optional) Resolution of heatmap.
            farm_opt (boolean, optional) Calls either farm or yaw optimisers.
        """

        # Wind speeds and turbulence intensities examined
        x_ws = np.linspace(3, 12, res)
        y_ti = np.linspace(0.05, 0.25, res)


        # Initialisation of power and timing heatmaps
        g0 = np.zeros((res, res))
        g1 = np.zeros((res, res))
        g2 = np.zeros((res, res))
        t1 = np.zeros((res, res))
        t2 = np.zeros((res, res))

        floris_park = wfct.floris_interface.FlorisInterface("FLORIS_input_gauss.json")

        # Begin parametric runs
        for k1 in range(res):

            # Print progress
            print(round(k1 / res * 100, 2), '%', 'complete.')

            for k2 in range(res):

                    g0[k1, k2] = abs(FLORIS_farm_power([0 for _ in range(len(xs))], xs, ys, x_ws[k1], y_ti[k2], floris_park))

                    opt_yaw, g1[k1, k2], t1[k1, k2] = FLORIS_wake_steering(xs, ys, [0 for _ in range(len(xs))], x_ws[k1],
                                                                       y_ti[k2], [-30, 30], 1.e-06, floris_path='./')

                    opt_yaw, g2[k1, k2], t2[k1, k2] = CNNwake_wake_steering(xs, ys, [0 for _ in range(len(xs))], x_ws[k1],
                                          y_ti[k2], CNN_generator, Power_model, TI_model,
                                          device, [-30, 30], 1.e-06)


        # Calculate FLORIS power gain in MW
        sample_1 = g1 - g0
        sample_2 = g2 - g0

        maxval = np.max([sample_1.min(), sample_1.max()])
        minval = np.min([sample_2.min(), sample_2.max()])


        makeHeatmap(np.transpose(np.flip(sample_1, 1)), x_ws, y_ti, maxval, minval, title='Floris optimisation')
        # Calculate FLORIS power gain in MW

        makeHeatmap(np.transpose(np.flip(sample_2, 1)), x_ws, y_ti, maxval, minval, title='Neural optimisation')

        # Calculate FLORIS average time
        sample_1 = t1
        sample_2 = t2

        maxval = np.max([sample_1.min(), sample_1.max()])
        minval = 0

        print('Average FLORIS time:', np.round(np.mean(t1), 2))
        makeHeatmap(np.transpose(np.flip(sample_1, 1)), x_ws, y_ti, maxval, minval, title='Floris time')
        # Calculate DNN average time
        print('Average DNN time:', np.round(np.mean(t2), 2))
        makeHeatmap(np.transpose(np.flip(sample_2, 1)), x_ws, y_ti, maxval, minval, title='Neural time')


    def makeHeatmap(bitmap, x_ws, y_ti, maxval, minval, title=None):
        """
        Plots bitmap of parametric optimisation runs.
        Args:
            bitmap (2D numpy array of floats) Calculated powers.
            x_ws (1D numpy array of floats) Wind speeds.
            y_ti (1D numpy array of floats) Turbulence intensities.
            vmax (float, optional) Max velocity cap of plot.
            title (string) Plot title.
        """

        # Min and max values of heatmap
        x_min = np.min(x_ws)
        x_max = np.max(x_ws)
        y_min = np.min(y_ti)
        y_max = np.max(y_ti)

        # Plot heatmap based on bitmap produced by the "Assess" function.
        plt.figure()
        plt.imshow(bitmap, cmap='RdYlGn', interpolation='nearest',
                   vmin=minval, vmax=maxval, extent=[x_min, x_max, y_min, y_max],
                   aspect=(x_max - x_min) / (y_max - y_min))

        plt.title(title, fontname='serif')
        plt.xlabel('Free stream velocity (m/s)', fontname='serif')
        plt.ylabel('Turbulence intensity', fontname='serif')

        plt.colorbar()
        plt.show()

    def power_map(xs, ys, u, ti, res=10):


        # Wind speeds and turbulence intensities examined
        yaw = np.linspace(0, 30, res)


        # Initialisation of power and timing heatmaps
        CNN_power_map = np.zeros((res, res))
        FLORIS_pwoer_map = np.zeros((res, res))

        floris_park = wfct.floris_interface.FlorisInterface("FLORIS_input_gauss.json")

        # Begin parametric runs
        for k1 in range(res):
            for k2 in range(res):
                FLORIS_pwoer_map[k1, k2] = abs(
                    FLORIS_farm_power([yaw[k1], yaw[k2]], xs, ys, u, ti, floris_park))/1.e6

                CNN_power_map[k1, k2] = abs(CNNWake_farm_power(
                                        [yaw[k1], yaw[k2]], xs, ys, u, ti,
                                        CNN_generator, Power_model, TI_model, device))/1.e6

        X, Y = np.meshgrid(yaw, yaw)

        # Twice as wide as it is tall.
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # ---- First subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.plot_surface(X, Y, FLORIS_pwoer_map, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('yaw 2')
        ax.set_ylabel('yaw 1')
        ax.set_zlabel('Power')
        ax.set_title('FLORIS')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.plot_surface(X, Y, CNN_power_map, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('yaw 2')
        ax.set_ylabel('yaw 1')
        ax.set_zlabel('Power')
        ax.set_title('CNNWAKE')

        plt.show()

    D = 121

    # Case B (yaw)
    xs = np.array([1 * D, 1 * D, 1 * D, 4.5 * D, 4.5 * D,
                   8 * D, 8 * D, 8 * D, 11.5 * D, 11.5 * D,
                   15 * D, 15 * D, 15 * D, 18.5 * D, 18.5 * D])
    ys = np.array([1 * D, 5 * D, 9 * D, 3 * D, 7 * D,
                   1 * D, 5 * D, 9 * D, 3 * D, 7 * D,
                   1 * D, 5 * D, 9 * D, 3 * D, 7 * D]) + 300

    #xs = np.array([1 * D, 1 * D, 8 * D, 8 * D, 15 * D, 15 * D])
    #ys = np.array([1 * D, 7 * D, 1 * D, 7 * D, 1 * D, 7 * D]) + 300

    #heatmap(xs, ys, res=8)


    power_map([300, 900], [500, 500], 9, 0.9, res=20)

