import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import matplotlib.pyplot as plt

# Constants for the problem setup
R = 2
num_points = 121  # Increased grid resolution for accuracy
M_size = num_points**2  # Size of the matrix M

# Calculate grid spacing
deltai = deltaj = 2 * R / num_points

# Function to calculate the discretized circle with boundary points
def calculate_circle_matrix_adjusted(num_points, R):
    x = np.linspace(-R, R, num_points)
    y = np.linspace(-R, R, num_points)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    circle_matrix = np.zeros((num_points, num_points))
    
    # Mark all points inside the circle
    for i in range(num_points):
        for j in range(num_points):
            if Z[i, j] <= R**2:
                circle_matrix[i, j] = 1

    # Adjust the boundary points
    for i in range(num_points):
        for j in range(num_points):
            if Z[i, j] > R**2 - deltai**2:
                circle_matrix[i, j] = 2

    return circle_matrix.flatten(), x, y

# Function to assemble the matrix M and vector b
def assemble_matrix(M_size, circle_matrix_flat, num_points, deltai):
    M = sp.dok_matrix((M_size, M_size), dtype=np.float64)
    b = np.zeros(M_size, dtype=np.float64)

    # Calculate the indices of the source
    source_i, source_j = (np.abs(x - 0.6)).argmin(), (np.abs(y - 1.2)).argmin()
    source_idx = source_i * num_points + source_j

    # Go through each point and set the corresponding equations
    for i in range(num_points):
        for j in range(num_points):
            idx = i * num_points + j
            if circle_matrix_flat[idx] == 1:  # Interior point
                if idx == source_idx:  # Source point
                    M[idx, idx] = 1
                    b[idx] = 1
                else:
                    # Use a five-point stencil for the Laplacian
                    M[idx, idx] = -4
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor_idx = (i + di) * num_points + (j + dj)
                        if circle_matrix_flat[neighbor_idx] != 0:
                            M[idx, neighbor_idx] = 1
                        else:
                            # Neighbor is outside the domain
                            b[idx] -= deltai**2  # Account for boundary condition c=0
            elif circle_matrix_flat[idx] == 2:  # Boundary point
                M[idx, idx] = 1
                b[idx] = 0  # Boundary condition c=0

    return M.tocsr(), b

# Main execution block
if __name__ == "__main__":
    circle_matrix_flat, x, y = calculate_circle_matrix_adjusted(num_points, R)
    M_csr, b = assemble_matrix(M_size, circle_matrix_flat, num_points, deltai)
    c = sp_linalg.spsolve(M_csr, b)  # Solve the system Mx = b
    
    # Reshape the solution for plotting
    c_reshaped = c.reshape((num_points, num_points))
    circle_matrix_reshaped = circle_matrix_flat.reshape((num_points, num_points))

    # Mask for points outside the circle and boundary
    outside_mask = circle_matrix_reshaped == 0
    boundary_mask = circle_matrix_reshaped == 2

    # Plot the steady-state concentration
    plt.figure(figsize=(8, 6))
    
    # Plot the concentration for the interior points with colormap
    concentration_plot = plt.contourf(x, y, np.ma.masked_where(outside_mask, c_reshaped), levels=50, cmap='viridis')

    # Plot white for outside the circle
    plt.contourf(x, y, np.ma.masked_where(~outside_mask, circle_matrix_reshaped), 1, colors='white')

    # Plot boundary in a specific color (let's say red)
    plt.contourf(x, y, np.ma.masked_where(~boundary_mask, circle_matrix_reshaped), 1, colors='white')

    # Create colorbar only for the concentration_plot
    plt.colorbar(concentration_plot, label='Concentration c')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Steady State Concentration')
    plt.axis('equal')
    plt.savefig('diffusion.png')
