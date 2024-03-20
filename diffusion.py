import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_linalg
import scipy.sparse as sp

R = 2
num_points = 20  # Increased for better resolution
D = 1
deltai = deltaj = 2 * R / num_points
M_size = num_points**2

def calculate_circle_matrix(num_points, R):
    # Discretize the space
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Initialize the circle matrix with zeros (outside the circle)
    circle_matrix = np.zeros((num_points, num_points))
    
    # Set points inside the circle to one and boundary points to two
    for i in range(num_points):
        for j in range(num_points):
            if Z[i, j] < R**2:
                circle_matrix[i, j] = 1
            elif Z[i, j] == R**2:
                circle_matrix[i, j] = 2
    
    return circle_matrix.flatten(), x, y  # Flatten the circle matrix

def index_in_flat_array(i, j, num_points):
    return i * num_points + j

def diff_solver(num_points, R):
    circle_matrix, x, y = calculate_circle_matrix(num_points, R)
    M = sp.dok_matrix((M_size, M_size))
    b = np.zeros(M_size)

    # Correctly handling the source index
    source_i = (np.abs(x - 0.6)).argmin()
    source_j = (np.abs(y - 1.2)).argmin()
    source_index = index_in_flat_array(source_i, source_j, num_points)
    b[source_index] = 1

    for i in range(num_points):
        for j in range(num_points):
            idx = index_in_flat_array(i, j, num_points)
            if circle_matrix[idx] == 1:  # Interior points
                M[idx, idx] = -4
                for di in [-1, 1]:
                    for dj in [-1, 1]:
                        if 0 <= i + di < num_points and 0 <= j + dj < num_points:
                            neighbor_idx = index_in_flat_array(i + di, j + dj, num_points)
                            if circle_matrix[neighbor_idx] == 1:
                                M[idx, neighbor_idx] = 1
                            else:
                                # Increment the diagonal value for each missing neighbor
                                M[idx, idx] += 1
            elif circle_matrix[idx] == 2:  # Boundary points
                M[idx, idx] = 1
                b[idx] = 0
            # For the source
            if idx == source_index:
                M[idx, idx] = 1

    # Convert to CSR format and solve
    M_csr = M.tocsr()
    c = sp_linalg.spsolve(M_csr, b)
    return c


def plot_steadystate(c, num_points, R):
    # Reshape c to the original grid shape and plot
    concentration = c.reshape((num_points, num_points))
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, concentration, levels=50)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    c = diff_solver(num_points, R)
    plot_steadystate(c, num_points, R)