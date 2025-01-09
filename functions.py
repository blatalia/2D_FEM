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

    return n, m, len_x, len_y, h_x, h_y, t0, tf, x, k1, k2, q_bottom, q_top

n, m, len_x, len_y, h_x, h_y, t0, tf, x, k1, k2, q_bottom, q_top = get_input_variables_from_file()
num_nodes_x = n + 1
num_nodes_y = m + 1
total_nodes = num_nodes_x * num_nodes_y


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

    def get_global_node(row, col):
        return row * num_nodes_x + col
    
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


# left Dirichlet boundary conditions
def dirichlet_left(global_matrix, load_vector, n, m, t_left):
    """
    apply Dirichlet boundary conditions on the left boundary (x = 0).
    """

    for j in range(num_nodes_y):
        node_index = j * num_nodes_x  # nodes on the left boundary
        load_vector[node_index] = t_left
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1

    return global_matrix, load_vector


# right Dirichlet boundary conditions
def dirichlet_right(global_matrix, load_vector, n, m, t_right):
    """
    apply Dirichlet boundary conditions on the right boundary (x = len_x).
    """

    for j in range(num_nodes_y):
        node_index = j * num_nodes_x + n  # nodes on the right boundary
        load_vector[node_index] = t_right
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1

    return global_matrix, load_vector


# Top Neumann boundary conditions
def neumann_top(load_vector, n, m, h_x, q_top):
    """
    apply Neumann boundary conditions on the top boundary (y = len_y).
    """
    for i in range(num_nodes_x):
        node_index = m * num_nodes_x + i  # nodes along the top boundary
        load_vector[node_index] += q_top * h_x  # contribution of heat flux over element's length

    return load_vector


# bottom Neumann boundary conditions
def neumann_bottom(load_vector, n, m, h_x, q_bottom):
    """
    apply Neumann boundary conditions on the bottom boundary (y = 0).
    """
    for i in range(num_nodes_x):
        node_index = i  # nodes along the bottom boundary
        load_vector[node_index] += q_bottom * h_x  # contribution of heat flux over element's length

    return load_vector

def solve_for_temperatures(global_matrix, load_vector):
    return np.linalg.solve(global_matrix, load_vector)

def get_node_coords(n, m, len_x, len_y):
    x_coords, y_coords = np.meshgrid(np.linspace(0, len_x, n + 1), np.linspace(0, len_y, m + 1))
    return x_coords, y_coords

def plot_nodes(x_coords, y_coords, len_x, len_y):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(y_coords, 0, len_x, color='black', linestyle='-', lw=0.6)
    ax.vlines(x_coords, 0, len_y, color='black', linestyle='-', lw=0.6)
    plt.plot(x_coords, y_coords, '.', color='black')
    
    count = 1
    for i,j in zip(x_coords, y_coords):
        for x,y in zip(i, j):
            ax.text(x, y, f'{count}', color='black', fontsize=10)
            # ax.text(x, y, f'({x:.2f}, {y:.2f}), {count}', color='black', fontsize=8)
            count += 1

    # to poniżej prostu nie działa z jakiegos powodu
    # for count, (x, y) in enumerate(zip(x_coords, y_coords), start=1):
    #     ax.text(x, y, f'{count}', color='black', fontsize=8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('nodes')
    plt.savefig('nodes.png')
    plt.show()

def plot_temp_at_node(temperatures, x_coords, y_coords, len_x, len_y):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(y_coords, 0, len_x, color='black', linestyle='-', lw=0.6)
    ax.vlines(x_coords, 0, len_y, color='black', linestyle='-', lw=0.6)
    plt.plot(x_coords, y_coords, '.', color='black')
    
    
    for i, temp in enumerate(temperatures, start=1):
        x = x_coords.flatten()[i-1]
        y = y_coords.flatten()[i-1]
        ax.text(x, y, f'{temp:.2f}', color='black', fontsize=8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('temperatures at nodes')
    plt.savefig('temp_at_node.png')
    plt.show()

def plot_temp_color_map(temperatures, x_coords, y_coords, len_x, len_y):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(y_coords, 0, len_x, color='black', linestyle='-', lw=0.3)
    ax.vlines(x_coords, 0, len_y, color='black', linestyle='-', lw=0.3)
    plt.plot(x_coords, y_coords, '.', color='black')
    
    plt.pcolormesh(x_coords, y_coords, temperatures.reshape(x_coords.shape))
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('temperatures at nodes')
    plt.savefig('temp_color_map.png')
    plt.show()

global_matrix = get_global_matrix(n, m, h_x, h_y, x, k1, k2)
print(global_matrix)

load_vector = np.zeros(global_matrix.shape[0])  # initialize load vector

global_matrix, load_vector = dirichlet_left(global_matrix, load_vector, n, m, t0)
global_matrix, load_vector = dirichlet_right(global_matrix, load_vector, n, m, tf)
load_vector = neumann_top(load_vector, n, m, h_x, q_top)
load_vector = neumann_bottom(load_vector, n, m, h_x, q_bottom)

print(load_vector)
print(global_matrix)

# save the modified global matrix and load vector
np.savetxt('global_matrix.txt', global_matrix)
np.savetxt('load_vector.txt', load_vector)

temperatures = solve_for_temperatures(global_matrix, load_vector)
print(temperatures)

# save the temperatures
np.savetxt('temperatures.txt', temperatures)

x_coords, y_coords = get_node_coords(n, m, len_x, len_y)
plot_nodes(x_coords, y_coords, len_x, len_y)

plot_temp_at_node(temperatures, x_coords, y_coords, len_x, len_y)
plot_temp_color_map(temperatures, x_coords, y_coords, len_x, len_y)
