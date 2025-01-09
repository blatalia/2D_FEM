import numpy as np
import matplotlib.pyplot as plt

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

    num_nodes_x = n + 1
    num_nodes_y = m + 1
    total_nodes = num_nodes_x * num_nodes_y
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

# dirichlet boundary conditions and updated global matrix
def dirichlet(global_matrix, n,m,t0,tf):
    # t0 - temperature on the left and right side of the plate
    # tf - temperature on the top and bottom side of the plate
    num_nodes_x = n + 1
    num_nodes_y = m + 1
    total_nodes = num_nodes_x * num_nodes_y
    F = np.zeros(total_nodes)
    for j in range(num_nodes_y):
        node_index = j * num_nodes_x
        F[node_index] = t0
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
    for j in range(num_nodes_y):
        node_index = j * num_nodes_x + n
        F[node_index] = tf
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
    
    return F, global_matrix

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
    plt.show()
    
# dane z wykładów
n = 20  
m = 20
len_x = 1.0
len_y = 1.0  
h_x = len_x / n  
h_y = len_y / m  
t0 = 100.0
tf = 270.0 
# x is where the material changes
x = 0.5
k1 = 0.1
k2 = 50

global_matrix = get_global_matrix(n, m, h_x, h_y, x, k1, k2)
print(global_matrix)

load_vector, global_matrix_updated = dirichlet(global_matrix, n,m,t0,tf)
print(load_vector)
print(global_matrix_updated)

temperatures = solve_for_temperatures(global_matrix_updated, load_vector)
print(temperatures)

x_coords, y_coords = get_node_coords(n, m, len_x, len_y)
plot_nodes(x_coords, y_coords, len_x, len_y)

plot_temp_at_node(temperatures, x_coords, y_coords, len_x, len_y)
plot_temp_color_map(temperatures, x_coords, y_coords, len_x, len_y)