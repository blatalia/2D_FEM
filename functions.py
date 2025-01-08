import numpy as np

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

def get_global_matrix(n, m, h_x, h_y, k):
    num_nodes_x = n + 1
    num_nodes_y = m + 1
    total_nodes = num_nodes_x * num_nodes_y
    K = np.zeros((total_nodes, total_nodes))

    def get_global_node(row, col):
        return row * num_nodes_x + col

    for elements_x in range(n):
        for elements_y in range(m):
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

def map_to_mesh(temperatures, n, m):
    num_nodes_x = n + 1
    num_nodes_y = m + 1
    T = np.zeros((num_nodes_y, num_nodes_x))
    for i in range(num_nodes_y):
        for j in range(num_nodes_x):
            T[i, j] = temperatures[i * num_nodes_x + j]
    return T

# dane z wykładów
n = 2  
m = 2  
h_x = 0.5  
h_y = 0.5  
k = 1.0  

global_matrix = get_global_matrix(n, m, h_x, h_y, k)
print(global_matrix)

load_vector, global_matrix_updated = dirichlet(global_matrix, n,m,0,1)
print(load_vector)
print(global_matrix_updated)

temperatures = solve_for_temperatures(global_matrix_updated, load_vector)
print(temperatures)

final_map = map_to_mesh(temperatures, n, m)
print(final_map)