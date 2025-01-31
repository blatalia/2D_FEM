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

