import numpy as np
import scipy.linalg
from tqdm import trange
import matplotlib.pyplot as plt
import time
from matplotlib.lines import Line2D
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigs

def define_matrix(N1, N2, dx):
    N12 = int(N1/dx)
    N22 = int(N2/dx)
    num_points = N12 * N22
    A = np.zeros((num_points, num_points))
    
    for i in trange(num_points):
        for j in range(num_points):
            if N12 == N22:
                if i not in range(0, N12) and i not in range(num_points-N12, num_points) and i % N12 != 0 and (i+1) % N12 != 0:
                    if i == j:
                        A[i, j] = -4 / (dx**2)
                    elif abs(i-j) == 1 or abs(i-j) == N12:
                        A[i, j] = 1 / (dx**2)
            else:
                if i not in range(0, N12) and i not in range(num_points-N12, num_points) and i % N22 != 0 and (i+1) % N22 != 0:

                    if i == j:
                        A[i, j] = -4 / (dx**2)
                    elif abs(i-j) == 1:
                        A[i, j] = 1 / (dx**2)
                    elif (i + N22) == j or (i - N22) == j:
                        A[i, j] = 1 / (dx**2)

    return A

def inside_circle(i,j,N12,N22,R,dx):
    x =  (i % N12) *dx - (N12*dx)/2
    y = (j // N12) *dx - (N22*dx)/2
    
    return x**2 + y**2 < R**2



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
    
    return circle_matrix.flatten()

def index_in_flat_array(i, j, num_points):
    return i * num_points + j

def A_matrix_circle(side, R):
    num_points = side**2
    dx = side/(2*R)
    A = np.zeros((num_points, num_points))
    circle = calculate_circle_matrix(side, R)
    
    for i in trange(num_points):
        for j in range(num_points):
            if circle[i] == 1:
                    if i == j:
                        A[i, j] = -4 / (dx**2)
                    elif abs(i - j) == 1 or abs(i - j) == side:
                        A[i, j] = 1 / (dx**2)
 
    return A
    
    

def plot_eigenmode(L, dx, type):
    if type == 'square':
        N1 = L
        N2 = L
        A = define_matrix(N1, N2, dx)
        N12 = int(N1/dx)
        N22 = int(N2/dx)
    elif type == 'rectangle':
        N1 = 2*L
        N2 = L
        A = define_matrix(N1, N2, dx)
        N12 = int(N1/dx)
        N22 = int(N2/dx)
    elif type == 'circle':
        A = A_matrix_circle(50,L)
        N1 = L
        N2 = L
        N12 = 50
        N22 = 50
    eig_values, eig_vectors = eigs(A, k=6, which='SR') # Compute the first six smallest real eigenvalues and eigenvectors

    for i in range(6):
        if type == 'circle' or type == 'square':
            fig, ax = plt.subplots(figsize=(5, 4)) 
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
        eig_vector_reshaped = eig_vectors[:, i].real.reshape(N12, N22)
        frequency = np.sqrt(np.abs(eig_values[i]))
        ax.imshow(eig_vector_reshaped, extent=(0, N1, 0, N2), cmap='winter', aspect='equal') 
        ax.set_title('$K$ = {:.2f}, $\lambda$ = {:.2f}'.format(eig_values[i].real, frequency))
        cbar = fig.colorbar(ax.imshow(eig_vector_reshaped, extent=(0, N1, 0, N2), cmap='winter', aspect='equal'), ax=ax)
        cbar.set_label('Eigenvector value', size=12)
        plt.savefig(f'eigenmode_{type}_{dx}_SR_{i+1}.png')  # Save each plot individually
        plt.close(fig)  # Close the figure after saving to free up memory

def compare_modes(L, dx, runs):
    A = define_matrix(L, L, dx)
    time_eigh = []
    time_eig = []
    time_eigs = []
    for i in range(runs):
        start = time.process_time()
        scipy.linalg.eigh(A)
        end = time.process_time()
        time_eigh.append(end - start)
        start = time.process_time()
        scipy.linalg.eig(A)
        end = time.process_time()
        time_eig.append(end - start)    
        start = time.process_time()
        eigs(A)
        end = time.process_time()
        time_eigs.append(end - start)

    average_time = [np.mean(time_eigh), np.mean(time_eig), np.mean(time_eigs)]
    stdev_time = [np.std(time_eigh), np.std(time_eig), np.std(time_eigs)]

    
    plt.figure(figsize=(10, 5))
    x_values = [1, 2, 3]
    methods = ['eigh', 'eig', 'eigs']

    for i in range(3):
        y_value = average_time[i]
        y_error = stdev_time[i]
        plt.errorbar(x_values[i], y_value, yerr=y_error, marker='o', capsize=10)

    plt.xticks(x_values, methods)
    plt.xlim(0.5, 3.5) 
    plt.ylabel('CPU Time (s)')
    plt.savefig(f'compare_modes_L_{L}_dx_{dx}_runs_{runs}.png')
    plt.show()
    

    

def difference_speed_sparse(L, dx, runs):
    A = define_matrix(L, L, dx)
    B = define_matrix(2*L, L, dx)
    circle = A_matrix_circle(int(L/dx),L)
    time_eigsA = []
    time_eigsB = []
    time_eigscircle = []
    time_sparsecircle = []
    time_sparseA = []
    time_sparseB = []
    for i in range(runs):
        start = time.process_time()
        eig_values, eig_vectors = eigs(A, k=3, which='SR')
        end = time.process_time()
        time_eigsA.append(end - start)
        start = time.process_time()
        eig_values_sparse, eig_vectors_sparse = scipy.sparse.linalg.eigs(A, k=3, which='SR')
        end = time.process_time()
        time_sparseA.append(end - start)
        start = time.process_time()
        eig_values, eig_vectors = eigs(B, k=3, which='SR')
        end = time.process_time()
        time_eigsB.append(end - start)
        start = time.process_time()
        eig_values_sparse, eig_vectors_sparse = scipy.sparse.linalg.eigs(B, k=3, which='SR')
        end = time.process_time()
        time_sparseB.append(end - start)
        start = time.process_time()
        eig_values, eig_vectors = eigs(circle, k=3, which='SR')
        end = time.process_time()
        time_eigscircle.append(end - start)
        start = time.process_time()
        eig_values_sparse, eig_vectors_sparse = scipy.sparse.linalg.eigs(circle, k=3, which='SR')
        end = time.process_time()
        time_sparsecircle.append(end - start)
        
    average_eigs = [np.mean(time_eigsA), np.mean(time_eigsB) , np.mean(time_eigscircle)]
    average_sparse = [np.mean(time_sparseA), np.mean(time_sparseB), np.mean(time_sparsecircle)]
    stdev_eigs = [np.std(time_eigsA), np.std(time_eigsB),   np.std(time_eigscircle)]
    stedv_sparse = [np.std(time_sparseA), np.std(time_sparseB), np.std(time_sparsecircle)]
    
    average_values = [average_eigs, average_sparse]
    stdev_values = [stdev_eigs, stedv_sparse]
    labels = ['Square', 'Rectangle', 'Circle']
    colors = ['red', 'blue', 'green']

    plt.figure(figsize=(10, 5))

    for i in range(3):
        y_values = [average_values[j][i] for j in range(2)]
        x_values = [1, 2]
        y_errors = [stdev_values[j][i] for j in range(2)]
        plt.errorbar(x_values, y_values, yerr=y_errors, marker='o', markerfacecolor=colors[i], markeredgecolor=colors[i], linestyle='', color=colors[i], capsize=10)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(3)]
    plt.legend(legend_elements, labels)

    plt.xticks(x_values, ['eigs', 'sparse'])
    plt.xlim(0.5, 2.5) 
    plt.ylabel('CPU Time (s)')
    plt.savefig(f'time_comparison_L_{L}_dx_{dx}_runs_{runs}.png')
    plt.show()

def compare_L(dx):
    L_values = [1, 2, 3, 4, 5]
    shapes = ['circle', 'square', 'rectangle']
    colors = ['red', 'blue', 'green']
    offsets = [-0.1, 0, 0.1]  

    plt.figure(figsize=(15, 5))

    for color, shape, offset in zip(colors, shapes, offsets):
        frequencies = []
        for L in L_values:
            if shape == 'circle':
                A = A_matrix_circle(int(L/dx), L)
            elif shape == 'square':
                A = define_matrix(L, L, dx)
            elif shape == 'rectangle':
                A = define_matrix(2*L, L, dx)
            eig_values, eig_vectors = eigs(A, k=10, which='SR')
            frequencies.append(np.sqrt(np.abs(eig_values)))

        for i, L in enumerate(L_values):
            y_values = frequencies[i]
            x_values = [L + offset]*len(y_values)  
            label = shape if i == 0 else None  
            plt.scatter(x_values, y_values, marker='o', color=color, label=label)

    plt.xlabel('L')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'compare_L_dx_{dx}.png')
    plt.show()



def compare_steps(L):
    steps = [0.1, 0.08, 0.06, 0.04, 0.02]
    shapes = ['circle', 'square', 'rectangle']
    colors = ['red', 'blue', 'green']
    offsets = [-0.001, 0, 0.001]  

    plt.figure(figsize=(15, 5))

    for color, shape, offset in zip(colors, shapes, offsets):
        frequencies = []
        for step in steps:
            if shape == 'circle':
                A = A_matrix_circle(int(L/step), L)
            elif shape == 'square':
                A = define_matrix(L, L, step)
            elif shape == 'rectangle':
                A = define_matrix(2*L, L, step)
            eig_values, eig_vectors = eigs(A, k=10, which='SR')
            frequencies.append(np.sqrt(np.abs(eig_values)))

        for i, step in enumerate(steps):
            y_values = frequencies[i]
            x_values = [step+offset]*len(y_values)  
            label = shape if i == 0 else None  
            plt.scatter(x_values, y_values, marker='o', color=color, label=label)

    plt.xlabel('Step size')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'compare_step_L_{L}.png')
    plt.show()
    
def time_dependent_solution(L, dx, T, dt, type):
    if type == 'square':
        N1 = L
        N2 = L
        A = define_matrix(N1, N2, dx)
        N12 = int(N1/dx)
        N22 = int(N2/dx)
    elif type == 'rectangle':
        N1 = 2*L
        N2 = L
        A = define_matrix(N1, N2, dx)
        N12 = int(N1/dx)
        N22 = int(N2/dx)
    elif type == 'circle':
        A = A_matrix_circle(50,L)
        N1 = L
        N2 = L
        N12 = 50
        N22 = 50
        
    eig_values, eig_vectors = eigs(A, k=4, which='SR')
    eig_vector_reshaped = eig_vectors[:, 0].real.reshape(N12, N22)  # Choose the first eigenmode
    omega = np.sqrt(np.abs(eig_values[0]))
    
    time_steps = int(T/dt)
    time = np.linspace(0, T, time_steps)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(np.arange(0, N12), np.arange(0, N22))
    ims = []

    def update_plot(t):
        Z = (np.cos(omega * time[t]) + np.sin(omega * time[t])) * eig_vector_reshaped
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='winter')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{type} Eigenmode 0 Time-Dependent Solution (t={time[t]:.2f})')

    ani = animation.FuncAnimation(fig, update_plot, frames=time_steps, interval=50)
    ani.save(f'{type}_eigenmode_0_animation.mp4', writer='ffmpeg')




def __main__():
    L = 1
    dx = 0.1
    runs = 20
    shapes = ['square', 'rectangle', 'circle']
    #compare_modes(L, dx, runs)
    #for i in shapes:
        #plot_eigenmode(L, dx, i)
    #difference_speed_sparse(L, dx, runs)
    compare_L(dx)
    #compare_steps(L)
    
__main__()
