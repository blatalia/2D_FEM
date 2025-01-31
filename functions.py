import numpy as np
import matplotlib.pyplot as plt
import json
import csv


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
    bc_top - boundary condition on the top side of the plate (Dirichlet or Neumann)
    bc_bottom - boundary condition on the bottom side of the plate (Dirichlet or Neumann)
    bc_left - boundary condition on the left side of the plate (Dirichlet or Neumann)
    bc_right - boundary condition on the right side of the plate (Dirichlet or Neumann)


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
    t_left = input_variables['t_left']
    t_right = input_variables['t_right']
    t_top = input_variables['t_top']
    t_bottom = input_variables['t_bottom']
    x = input_variables['x']
    k1 = input_variables['k1']
    k2 = input_variables['k2']
    q_right = input_variables['q_right']
    q_left = input_variables['q_left']
    q_bottom = input_variables['q_bottom']
    q_top = input_variables['q_top']
    bc_top = input_variables['bc_top']
    bc_bottom = input_variables['bc_bottom']
    bc_left = input_variables['bc_left']
    bc_right = input_variables['bc_right']
    xy_plot = input_variables['xy_plot']
    point = input_variables['point']
    heat_value = input_variables['heat_value']

    return n, m, len_x, len_y, h_x, h_y, t_left, t_right, t_top, t_bottom, x, k1, k2, q_right, q_left, q_bottom, q_top, bc_top, bc_bottom, bc_left, bc_right, xy_plot, point, heat_value

def get_local_matrix(h_x, h_y, k):
    """
    Function to get the local matrix for a 4-node quadrilateral element.

    :param h_x: length of the element in the x direction
    :param h_y: length of the element in the y direction
    :param k: thermal conductivity of the material
    :return: local matrix for the element
    """
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


def get_global_matrix(n, m, h_x, h_y, x, k1, k2, total_nodes, num_nodes_x):
    """
    Function to get the global matrix for the 2D domain.

    :param n: number of elements in the x direction
    :param m: number of elements in the y direction
    :param h_x: length of the element in the x direction
    :param h_y: length of the element in the y direction
    :param x: where the material changes
    :param k1: thermal conductivity of the material on the left side of the plate
    :param k2: thermal conductivity of the material on the right side of the plate
    :param total_nodes: total number of nodes
    :param num_nodes_x: number of nodes in the x direction
    :return: global matrix for the 2D domain
    """
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


def dirichlet_left(global_matrix, load_vector, num_nodes_x, num_nodes_y, t_left):
    """
    apply Dirichlet boundary conditions on the left boundary (x = 0).
    """

    for j in range(num_nodes_y):
        node_index = j * num_nodes_x #was: num_nodes_y, changed to num_nodes_x 
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
        load_vector[node_index] = t_left

    return global_matrix, load_vector


def dirichlet_right(global_matrix, load_vector, n, num_nodes_x, num_nodes_y, t_right):
    """
    apply Dirichlet boundary conditions on the right boundary (x = len_x).
    """
    
    for j in range(num_nodes_y):
        node_index = j * num_nodes_x + n #was: num_nodes_y, changed to num_nodes_x
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
        load_vector[node_index] = t_right

    return global_matrix, load_vector


def dirichlet_bottom(global_matrix, load_vector, num_nodes_x, t_bottom):
    """
    apply Dirichlet boundary conditions on the bottom boundary (y = 0).
    """    
    
    for i in range(num_nodes_x):
        node_index = i 
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
        load_vector[node_index] = t_bottom

    return global_matrix, load_vector


def dirichlet_top(global_matrix, load_vector, m, num_nodes_x, t_top):
    """
    apply Dirichlet boundary conditions on the top boundary (y = len_y).
    """ 
    
    for i in range(num_nodes_x):
        node_index = m * num_nodes_x + i  
        global_matrix[node_index, :] = 0
        global_matrix[node_index, node_index] = 1
        load_vector[node_index] = t_top

    return global_matrix, load_vector


def neumann_left(load_vector, num_nodes_x, num_nodes_y, h_y, q_left):
    """
    apply Neumann boundary conditions on the left boundary (x = 0).
    """    
    
    for j in range(num_nodes_y):
        node_index = j * num_nodes_x
        load_vector[node_index] += q_left * h_y

    return load_vector


def neumann_right(load_vector, n, num_nodes_x, num_nodes_y, h_y, q_right):
    """
    apply Neumann boundary conditions on the right boundary (x = len_x).
    """    
    
    for j in range(num_nodes_y):
        node_index = j * num_nodes_x + n
        load_vector[node_index] += q_right * h_y

    return load_vector


def neumann_top(load_vector, m, num_nodes_x, h_x, q_top):
    """
    apply Neumann boundary conditions on the top boundary (y = len_y).
    """
        
    for i in range(num_nodes_x):
        node_index = (m * num_nodes_x) + i
        load_vector[node_index] += q_top * h_x

    return load_vector


def neumann_bottom(load_vector, num_nodes_x, h_x, q_bottom):
    """
    apply Neumann boundary conditions on the bottom boundary (y = 0).
    """  
    
    for i in range(num_nodes_x):
        node_index = i
        load_vector[node_index] += q_bottom * h_x

    return load_vector


def solve_for_temperatures(global_matrix, load_vector):
    """
    solve for the temperatures at nodes
    """
    return np.linalg.solve(global_matrix, load_vector)

def get_node_coords(n, m, len_x, len_y):
    """
    get node coordinates
    """
    x_coords, y_coords = np.meshgrid(np.linspace(0, len_x, n + 1), np.linspace(0, len_y, m + 1))
    return x_coords, y_coords

def plot_nodes(x_coords, y_coords, len_x, len_y):
    """
    plot the mesh with nodes enumerated
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(y_coords, 0, len_x, color='black', linestyle='-', lw=0.6)
    ax.vlines(x_coords, 0, len_y, color='black', linestyle='-', lw=0.6)
    plt.plot(x_coords, y_coords, '.', color='black')
    
    count = 1
    for i,j in zip(x_coords, y_coords):
        for x,y in zip(i, j):
            ax.text(x, y, f'{count}', color='black', fontsize=10)
            count += 1

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Nodes')
    plt.savefig('nodes.png')
    plt.show()

def plot_temp_at_node(temperatures, x_coords, y_coords, len_x, len_y):
    """
    plot the temperatures at given nodes and save temperatures with appropriate coordinates to a file
    """
    fig = plt.figure(figsize=(20, 15))  
    ax = fig.add_subplot()
    ax.hlines(y_coords, 0, len_x, color='black', linestyle='-', lw=0.6)
    ax.vlines(x_coords, 0, len_y, color='black', linestyle='-', lw=0.6)
    plt.plot(x_coords, y_coords, '.', color='black')
    
    temp_data = [] 

    for i, t in enumerate(temperatures, start=1):  
        x = x_coords.flatten()[i-1]
        y = y_coords.flatten()[i-1]
        ax.text(x, y, f'{t:.2f} K', color='black', fontsize=12)
        temp_data.append([round(x, 4), round(y, 4), t])  

    with open('temp_at_coords.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'temperature [K]'])
        writer.writerows(temp_data)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Temperatures at Nodes')
    plt.savefig('temp_at_node.png')
    plt.show()


def plot_temp_color_mesh(temperatures, x_coords, y_coords, len_x, len_y):
    """
    plot the node temperatures and the mesh
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hlines(y_coords, 0, len_x, color='black', linestyle='-', lw=0.3)
    ax.vlines(x_coords, 0, len_y, color='black', linestyle='-', lw=0.3)
    plt.plot(x_coords, y_coords, '.', color='black')
    
    plt.pcolormesh(x_coords, y_coords, temperatures.reshape(x_coords.shape))
    plt.colorbar(label='Temperature [K]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Temperatures at Nodes')
    plt.savefig('temp_color_mesh.png')
    plt.show()

def plot_temp_color_map(temperatures, x_coords, y_coords, len_x, len_y):
    """
    plot the node temperatures without the mesh
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    plt.pcolormesh(x_coords, y_coords, temperatures.reshape(x_coords.shape))
    plt.colorbar(label='Temperature [K]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Temperatures at Nodes')
    plt.savefig('temp_color_map.png')
    plt.show()



n, m, len_x, len_y, h_x, h_y, t_left, t_right, t_top, t_bottom, x, k1, k2, q_right, q_left, q_bottom, q_top, bc_top, bc_bottom, bc_left, bc_right, xy_plot, point, heat_value = get_input_variables_from_file()

def bc_and_heat_source(n, m, len_x, len_y, h_x, h_y, t_left, t_right, t_top, t_bottom, x, k1, k2, q_right, q_left, q_bottom, q_top, bc_top, bc_bottom, bc_left, bc_right, point, heat_value):
    """
    Function to first gather the input variables from input.json,
    then apply the boundary conditions, and finally solve for the temperatures.

    :return:
    global_matrix: modified global matrix after applying boundary conditions
    load_vector: modified load vector after applying boundary conditions
    temperatures: array of temperatures at nodes
    x_coords: 2D array of x-coordinates of nodes
    y_coords: 2D array of y-coordinates of nodes
    len_x: length of the plate in the x direction
    len_y: length of the plate in the y direction
    num_nodes_x: number of nodes in the x direction
    num_nodes_y: number of nodes in the y direction
    total_nodes: total number of nodes
    """

    num_nodes_x = n + 1
    num_nodes_y = m + 1
    total_nodes = num_nodes_x * num_nodes_y

    global_matrix = get_global_matrix(n, m, h_x, h_y, x, k1, k2, total_nodes, num_nodes_x)
    load_vector = np.zeros(global_matrix.shape[0]) 

    if bc_top == 'Dirichlet':
        global_matrix, load_vector = dirichlet_top(global_matrix, load_vector, m, num_nodes_x, t_top)
    elif bc_top == 'Neumann':
        load_vector = neumann_top(load_vector, m, num_nodes_x, h_x, q_top)

    if bc_bottom == 'Dirichlet':
        global_matrix, load_vector = dirichlet_bottom(global_matrix, load_vector, num_nodes_x, t_bottom)
    elif bc_bottom == 'Neumann':
        load_vector = neumann_bottom(load_vector, num_nodes_x, h_x, q_bottom)

    if bc_left == 'Dirichlet':
        global_matrix, load_vector = dirichlet_left(global_matrix, load_vector, num_nodes_x, num_nodes_y, t_left)
    elif bc_left == 'Neumann':
        load_vector = neumann_left(load_vector, num_nodes_x, num_nodes_y, h_y, q_left)

    if bc_right == 'Dirichlet':
        global_matrix, load_vector = dirichlet_right(global_matrix, load_vector, n, num_nodes_x, num_nodes_y, t_right)
    elif bc_right == 'Neumann':
        load_vector = neumann_right(load_vector, n, num_nodes_x, num_nodes_y, h_y, q_right)

    x_coords, y_coords = get_node_coords(n, m, len_x, len_y)
    
    node_positions = np.array([(x, y) for x_row, y_row in zip(x_coords, y_coords) for x, y in zip(x_row, y_row)])
    node_count = len(node_positions)

    if point is not None:
        if 0 <= point[0] <= len_x and 0 <= point[1] <= len_y:
            distances = np.linalg.norm(node_positions - np.array(point), axis=1)
            nearest_node = np.argmin(distances) + 1

            load_vector[nearest_node - 1] += heat_value
        else:
            print('Point is out of material range. It will not be taken into account.')

    temperatures = solve_for_temperatures(global_matrix, load_vector)

    return global_matrix, load_vector, temperatures, x_coords, y_coords, num_nodes_x, num_nodes_y, total_nodes


def interpolate_temperature_2d(temperatures, x_coords, y_coords, num_points=10):
    """
    Function to interpolate temperatures in 2D domain.

    :param temperatures: Flattened array of temperatures at nodes
    :param x_coords: 2D array of x-coordinates of nodes
    :param y_coords: 2D array of y-coordinates of nodes
    :param num_points: Number of interpolation points between each node
    """
    num_nodes_x = x_coords.shape[1]
    num_nodes_y = y_coords.shape[0]

    fine_x = np.linspace(0, x_coords.max(), (num_nodes_x - 1) * num_points + 1)
    fine_y = np.linspace(0, y_coords.max(), (num_nodes_y - 1) * num_points + 1)
    fine_xx, fine_yy = np.meshgrid(fine_x, fine_y)
    fine_temperatures = np.zeros_like(fine_xx)

    for i in range(num_nodes_y - 1):
        for j in range(num_nodes_x - 1):
            n1 = i * num_nodes_x + j
            n2 = n1 + 1
            n3 = n1 + num_nodes_x + 1
            n4 = n1 + num_nodes_x

            x1, y1, t1 = x_coords[i, j], y_coords[i, j], temperatures[n1]
            x2, y2, t2 = x_coords[i, j + 1], y_coords[i, j + 1], temperatures[n2]
            x3, y3, t3 = x_coords[i + 1, j + 1], y_coords[i + 1, j + 1], temperatures[n3]
            x4, y4, t4 = x_coords[i + 1, j], y_coords[i + 1, j], temperatures[n4]

            def shape_functions(x, y):
                denom = (x2 - x1) * (y3 - y1)
                N1 = ((x2 - x) * (y3 - y)) / denom
                N2 = ((x - x1) * (y3 - y)) / denom
                N3 = ((x - x1) * (y - y1)) / denom
                N4 = ((x2 - x) * (y - y1)) / denom
                return N1, N2, N3, N4

            x_fine = fine_x[j * num_points:(j + 1) * num_points + 1]
            y_fine = fine_y[i * num_points:(i + 1) * num_points + 1]
            xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)

            for xi, yi in zip(xx_fine.flatten(), yy_fine.flatten()):
                if xi > x2: xi = x2
                if yi > y3: yi = y3
                N1, N2, N3, N4 = shape_functions(xi, yi)
                temp = N1 * t1 + N2 * t2 + N3 * t3 + N4 * t4

                ix = min(np.searchsorted(fine_x, xi), len(fine_x) - 1)
                iy = min(np.searchsorted(fine_y, yi), len(fine_y) - 1)
                fine_temperatures[iy, ix] = temp

    fine_temperatures[:, -1] = np.interp(fine_y, y_coords[:, -1], temperatures[num_nodes_x - 1::num_nodes_x])  # Right edge
    fine_temperatures[-1, :] = np.interp(fine_x, x_coords[-1, :], temperatures[-num_nodes_x:])  # Top edge

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(fine_xx, fine_yy, fine_temperatures, shading='auto', cmap='rainbow')
    plt.colorbar(label='Temperature [K]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Temperature Distribution Across 2D Domain')
    plt.savefig('temperature_distribution_2d.png')
    plt.show()



def interpolate_temperature_along_line(temperatures, x_coords, y_coords, line='horizontal', position=0.5, num_points=10):
    """
    Interpolate temperatures along a specific line using shape functions, with more points for smoother interpolation.

    :param temperatures: Flattened array of temperatures at nodes
    :param x_coords: 2D array of x-coordinates of nodes
    :param y_coords: 2D array of y-coordinates of nodes
    :param line: 'horizontal' or 'vertical' line
    :param position: Relative position of the line (0 to 1), e.g., 0.5 for centerline
    :param num_points: Number of interpolation points between each node
    """
    num_nodes_x = x_coords.shape[1]
    num_nodes_y = y_coords.shape[0]

    if line == 'horizontal':
        y_pos = position * y_coords.max()
        row_index = int(position * (num_nodes_y - 1))
        x_line = x_coords[row_index, :]
        temps_line = temperatures[row_index * num_nodes_x:(row_index + 1) * num_nodes_x]
    elif line == 'vertical':
        x_pos = position * x_coords.max()
        col_index = int(position * (num_nodes_x - 1))
        y_line = y_coords[:, col_index]
        temps_line = temperatures[col_index::num_nodes_x]

    interpolated_x = []
    interpolated_temps = []

    for i in range(len(temps_line) - 1):
        x_start = x_line[i] if line == 'horizontal' else y_line[i]
        x_end = x_line[i + 1] if line == 'horizontal' else y_line[i + 1]
        temp_start = temps_line[i]
        temp_end = temps_line[i + 1]

        for j in range(num_points + 1):
            xi = j / num_points
            N1 = 1 - xi  
            N2 = xi      
            interpolated_temp = N1 * temp_start + N2 * temp_end
            interpolated_pos = N1 * x_start + N2 * x_end

            interpolated_x.append(interpolated_pos)
            interpolated_temps.append(interpolated_temp)

    plt.figure(figsize=(10, 6))

    if line == 'horizontal':
        plt.plot(interpolated_x, interpolated_temps, '-o', label=f'Temperature along y={y_pos:.2f} m')
        plt.xlabel('x [m]')
    elif line == 'vertical':
        plt.plot(interpolated_x, interpolated_temps, '-o', label=f'Temperature along x={x_pos:.2f} m')
        plt.xlabel('y [m]')

    plt.ylabel('Temperature [K]')
    plt.legend()
    plt.grid(True)
    plt.title('Temperature Distribution Along Line')
    plt.savefig('temp_along_line.png')
    plt.show()



global_matrix, load_vector, temperatures, x_coords, y_coords, num_nodes_x, num_nodes_y, total_nodes = bc_and_heat_source(n, m, len_x, len_y, h_x, h_y, t_left, t_right, t_top, t_bottom, x, k1, k2, q_right, q_left, q_bottom, q_top, bc_top, bc_bottom, bc_left, bc_right, point, heat_value)
# print(load_vector)
# print(global_matrix)
# print(temperatures)

plot_nodes(x_coords, y_coords, len_x, len_y)
plot_temp_at_node(temperatures, x_coords, y_coords, len_x, len_y)
plot_temp_color_mesh(temperatures, x_coords, y_coords, len_x, len_y)
plot_temp_color_map(temperatures, x_coords, y_coords, len_x, len_y)
interpolate_temperature_2d(temperatures, x_coords, y_coords, num_points=20)
interpolate_temperature_along_line(temperatures, x_coords, y_coords, line=xy_plot, position=0.5, num_points=20)