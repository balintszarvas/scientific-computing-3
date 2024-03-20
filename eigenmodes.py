import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import scipy.linalg
from tqdm import trange
import matplotlib.pyplot as plt

# Define the matrix A
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



def plot_eigenmode(N1, N2, dx):
    A = define_matrix(N1, N2, dx)
    N12 = int(N1/dx)
    N22 = int(N2/dx)
    eig_values, eig_vectors = eigs(A, k=3, which='LM') # Compute the first three smallest magnitude eigenvalues and eigenvectors
    fig, axs = plt.subplots(1, 3, figsize=(15, 4)) 
    for i in range(3):
        eig_vector_reshaped = eig_vectors[:, i].real.reshape(N12, N22)
        frequency = np.sqrt(np.abs(eig_values[i]))
        if i == 0:
            axs[i].imshow(eig_vector_reshaped, extent=(0, N1, 0, N2), cmap='winter', aspect='equal') 
            axs[i].set_title('$K$ = {:.2f}, $\omega$ = {:.2f}'.format(eig_values[i].real, frequency))
        else:
            axs[i].imshow(eig_vector_reshaped, extent=(0, N1, 0, N2),  cmap='winter', aspect='equal') 
            axs[i].set_title('$K$ = {:.2f}, $\omega$ = {:.2f}'.format(eig_values[i].real, frequency))
            axs[i].set_yticks([])  
            axs[i].set_ylabel('')  
    
    cbar = fig.colorbar(axs[2].imshow(eig_vector_reshaped, extent=(0, N1, 0, N2), cmap='winter', aspect='equal'), ax=axs[2])
    cbar.set_label('Eigenvector value', size=12) 
    
    plt.tight_layout()
    if N12 == N22:
        plt.savefig(f'eigenmode_square_{dx}_LM.png')  
    else:
        plt.savefig(f'eigenmode_rectangle_{dx}_LM.png')
    plt.show()




#Compare de modes to compute eigenvalues and eigenvectors - eig ; eigs; eigh

def __main__():
    N1 = 2 #increase this one
    N2 = 1
    dx = 0.01
    plot_eigenmode(N1, N2, dx)
    
__main__()
