import numpy as np
import matplotlib.pyplot as plt
import json

def get_input_variables_from_file(file_path: str = 'input.json'):
    """
    Function to read input variables from a JSON file. The file should contain the following variables:
    n - number of elements in the x direction
    m - number of elements in the y direction
    len_x - length of the plate in the x direction
    len_y - length of the plate in the y direction
    t0 - temperature on the left and right side of the plate
    tf - temperature on the top and bottom side of the plate
    x - where the material changes
    k1 - thermal conductivity of the material on the left side of the plate
    k2 - thermal conductivity of the material on the right side of the plate
    q - heat flux on the top side of the plate
    point_sources - list of point sources with their positions and values
    volumetric_sources - list of volumetric sources with their positions and values
    openings - list of openings with their positions

    :param file_path: path to the file containing the input variables. Default is 'input.json'.
    :return input_variables: dictionary containing the input variables. If the file is not found, a FileNotFoundError is raised.
    """

    try:
        with open(file_path, 'r') as file:
            input_variables = json.load(file)

            n = input_variables['n']
            m = input_variables['m']
            len_x = input_variables['len_x']
            len_y = input_variables['len_y']

            input_variables['h_x'] = len_x / n
            input_variables['h_y'] = len_y / m

    except FileNotFoundError:
        raise FileNotFoundError(f'File {file_path} not found')

    n = input_variables['n']
    m = input_variables['m']
    len_x = input_variables['len_x']
    len_y = input_variables['len_y']
    h_x = input_variables['h_x']
    h_y = input_variables['h_y']
    t0 = input_variables['t0']
    tf = input_variables['tf']
    x = input_variables['x']
    k1 = input_variables['k1']
    k2 = input_variables['k2']
    q_bottom = input_variables['q_bottom']
    q_top = input_variables['q_top']
    point_sources = input_variables.get('point_sources', [])
    volumetric_sources = input_variables.get('volumetric_sources', [])
    openings = input_variables.get('openings', [])

    return n, m, len_x, len_y, h_x, h_y, t0, tf, x, k1, k2, q_bottom, q_top, point_sources, volumetric_sources, openings

n, m, len_x, len_y, h_x, h_y, t0, tf, x, k1, k2, q_bottom, q_top, point_sources, volumetric_sources, openings = get_input_variables_from_file()
num_nodes_x = n + 1
num_nodes_y = m + 1
total_nodes = num_nodes_x * num_nodes_y

def get_global_node(row, col):
    return row * num_nodes_x + col

def apply_point_sources(load_vector, point_sources, h_x, h_y):
    for source in point_sources:
        row, col, value = source
        node_index = get_global_node(row, col)
        load_vector[node_index] += value
    return load_vector

def apply_volumetric_sources(load_vector, volumetric_sources, h_x, h_y):
    for source in volumetric_sources:
        row_start, col_start, row_end, col_end, value = source
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                node_index = get_global_node(row, col)
                load_vector[node_index] += value * h_x * h_y
    return load_vector

def apply_openings(global_matrix, load_vector, openings):
    for opening in openings:
        row, col = opening
        node_index = get_global_node(row, col)
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
        load_vector[node_index] = 0
    return global_matrix, load_vector

def get_local_matrix(h_x, h_y, k):
    K11 = (k/(3*h_x*h_y))*(h_y**2 + h_x**2)
    K12 = (k/(6*h_x*h_y))*(h_x**2 - 2*h_y**2)
    K13 = -(k/(6*h_x*h_y))*(h_x**2 + h_y**2)
    K14 = -(k/(6*h_x*h_y))*(2*h_x**2 - h_y**2)
    K21 = (k/(6*h_x*h_y))*(h_x**2 - 2*h_y**2)
    K22 = (k/(3*h_x*h_y))*(h_y**2 + h_x**2)
    K23 = -(k/(6*h_x*h_y))*(2*h_x**2 - h_y**2)
    K24 = -(k/(6*h_x*h_y))*(h_x**2 + h_y**2)
    K31 = -(k/(6*h_x*h_y))*(h_x**2 + h_y**2)
    K32 = -(k/(6*h_x*h_y))*(2*h_x**2 - h_y**2)
    K33 = (k/(3*h_x*h_y))*(h_y**2 + h_x**2)
    K34 = (k/(6*h_x*h_y))*(h_x**2 - 2*h_y**2)
    K41 = -(k/(6*h_x*h_y))*(2*h_x**2 - h_y**2)
    K42 = -(k/(6*h_x*h_y))*(h_x**2 + h_y**2)
    K43 = (k/(6*h_x*h_y))*(h_x**2 - 2*h_y**2)
    K44 = (k/(3*h_x*h_y))*(h_y**2 + h_x**2)
    return np.array([[K11, K12, K13, K14], [K21, K22, K23, K24], [K31, K32, K33, K34], [K41, K42, K43, K44]])

def get_global_matrix(n, m, h_x, h_y, x, k1, k2):
    if k2 == None:
        k2 = k1

    K = np.zeros((total_nodes, total_nodes))

    for elements_x in range(n):
        for elements_y in range(m):
            element_x_center = (elements_x + 0.5) * h_x
            if element_x_center < x:
                k = k1
            else:
                k = k2
            local_matrix = get_local_matrix(h_x, h_y, k)

            n1 = get_global_node(elements_y, elements_x)
            n2 = get_global_node(elements_y, elements_x + 1)
            n3 = get_global_node(elements_y + 1, elements_x + 1)
            n4 = get_global_node(elements_y + 1, elements_x)
            global_indices = [n1, n2, n3, n4]

            for i in range(4):
                for j in range(4):
                    K[global_indices[i], global_indices[j]] += local_matrix[i, j]

    return K

def solve_for_temperatures(global_matrix, load_vector):
    return np.linalg.solve(global_matrix, load_vector)

load_vector = np.zeros(total_nodes)

global_matrix = get_global_matrix(n, m, h_x, h_y, x, k1, k2)
load_vector = apply_point_sources(load_vector, point_sources, h_x, h_y)
load_vector = apply_volumetric_sources(load_vector, volumetric_sources, h_x, h_y)
global_matrix, load_vector = apply_openings(global_matrix, load_vector, openings)

temperatures = solve_for_temperatures(global_matrix, load_vector)
print(temperatures)

np.savetxt('temperatures.txt', temperatures)

def get_temperature_along_line(temperatures, x_coords, y_coords, line_type='horizontal', line_position=0.5):
    """
    Function to extract temperatures along a specific line in the 2D domain.

    Parameters:
    - temperatures: 1D array of temperatures at each node.
    - x_coords: 2D array of x-coordinates of the nodes.
    - y_coords: 2D array of y-coordinates of the nodes.
    - line_type: 'horizontal' or 'vertical' (default is 'horizontal').
    - line_position: normalized position of the line (0 to 1), where 0 is the bottom/left and 1 is the top/right.

    Returns:
    - line_coords: 1D array of x or y coordinates along the line.
    - line_temps: 1D array of temperatures along the line.
    """
    if line_type == 'horizontal':
        row_idx = int(line_position * (y_coords.shape[0] - 1))
        line_coords = x_coords[row_idx, :]
        line_temps = temperatures.reshape(y_coords.shape)[row_idx, :]
    elif line_type == 'vertical':
        col_idx = int(line_position * (x_coords.shape[1] - 1))
        line_coords = y_coords[:, col_idx]
        line_temps = temperatures.reshape(x_coords.shape)[:, col_idx]
    else:
        raise ValueError("line_type must be 'horizontal' or 'vertical'")

    return line_coords, line_temps

def plot_temperature_along_line(line_coords, line_temps, line_type='horizontal'):
    """
    Function to plot the temperature distribution along a specific line.

    Parameters:
    - line_coords: 1D array of coordinates along the line.
    - line_temps: 1D array of temperatures along the line.
    - line_type: 'horizontal' or 'vertical' (default is 'horizontal').
    """
    plt.figure()
    plt.plot(line_coords, line_temps, marker='o')
    plt.xlabel('x' if line_type == 'horizontal' else 'y')
    plt.ylabel('temperature (°C)')
    plt.title(f'temperature distribution along {line_type} line')
    plt.grid(True)
    plt.savefig(f'temp_along_{line_type}_line.png')
    plt.show()

x_coords, y_coords = np.meshgrid(np.linspace(0, len_x, n + 1), np.linspace(0, len_y, m + 1))
x_line_coords, x_line_temps = get_temperature_along_line(temperatures, x_coords, y_coords, line_type='horizontal', line_position=0.5)
plot_temperature_along_line(x_line_coords, x_line_temps, line_type='horizontal')
y_line_coords, y_line_temps = get_temperature_along_line(temperatures, x_coords, y_coords, line_type='vertical', line_position=0.5)
plot_temperature_along_line(y_line_coords, y_line_temps, line_type='vertical')
