import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

#contants
m = 1
k_values = [1, 2, 5]
s_values = [0, 1, 1.5, 4]
x0 = 1
v0 = 0
t = np.linspace(0, 5, 1000)

dt = t[1] - t[0]  # Time step size

def leapfrog(k, m, x0, v0, t):
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    a = np.zeros(len(t))
    x[0] = x0
    v[0] = v0

    a[0] = -k * x[0] / m

    for i in range(0, len(t) - 1):
        if i == 0:
            v_half = v[0] + 0.5 * a[0] * dt
        else:
            v_half += a[i] * dt
        
        x[i + 1] = x[i] + v_half * dt
        a[i + 1] = -k * x[i + 1] / m
        v[i + 1] = v_half + 0.5 * a[i + 1] * dt
    return t, x, v

def leapfrog_sin(k, m, x0, v0, t, s):
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    a = np.zeros(len(t))
    x[0] = x0
    v[0] = v0

    a[0] = -k * x[0] / m

    for i in range(0, len(t) - 1):
        if i == 0:
            v_half = v[0] + 0.5 * a[0] * dt
        else:
            v_half += a[i] * dt
        
        x[i + 1] = x[i] + v_half * dt
        a[i + 1] = np.sin(t[i]*s)-k * x[i + 1] / m
        v[i + 1] = v_half + 0.5 * a[i + 1] * dt
    return t, x, v

def plot_leapfrog(k_values, m, x0, v0, t):
    figs, ax = plt.subplots(1, 2, figsize=(15, 5))
    for k in k_values:
        _, x, v = leapfrog(k, m, x0, v0, t)
        ax[0].plot(t, x, label=f'k = {k}')
        ax[1].plot(t, v, label=f'k = {k}')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Position')
        ax[1].set_xlabel('Time')
        ax[0].legend()
        ax[1].legend()
        ax[1].set_ylabel('Velocity')
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].set_title('Position')
        ax[1].set_title('Velocity')
    plt.tight_layout()
    plt.savefig('leapfrog.png')

def plot_phase(s_values, k):
    fig = plt.figure()
    for i, s in enumerate(s_values):
        _, xl, vl = leapfrog(k, m, x0, v0, t)
        _, x, v = leapfrog_sin(k, m, x0, v0, t, s)
        plt.plot(x, v, label=f's = {s}')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('phase.png')

def runge_kutta(k, m, x0, v0, t_end):
    def f(t, y):
        return [y[1], -k * y[0] / m]
    
    t_values = [0]
    x_values = [x0]
    v_values = [v0]

    sol = RK45(f, t_values[0], [x0, v0], t_end)

    while sol.status == 'running':
        sol.step()
        t_values.append(sol.t)
        x_values.append(sol.y[0])
        v_values.append(sol.y[1])

    return np.array(t_values), np.array(x_values), np.array(v_values)
    
def calculate_energy(x, v, k, m):
    return 0.5 * m * v**2 + 0.5 * k * x**2

def compare_energy(k_values, m, x0, v0, t):
    plt.figure(figsize=(10, 6))

    for k in k_values:
        t_leap, x_leap, v_leap = leapfrog(k, m, x0, v0, t)
        energy_leap = calculate_energy(x_leap, v_leap, k, m)
        plt.plot(t_leap, energy_leap, label=f'Leapfrog at k = {k}')
        
        t_rk45, x_rk45, v_rk45 = runge_kutta(k, m, x0, v0, t[-1])
        energy_rk45 = calculate_energy(x_rk45, v_rk45, k, m)
        plt.plot(t_rk45, energy_rk45, label=f'RK45 at k = {k}', linestyle='--')

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Comparison between Leapfrog and RK45')
    plt.savefig('energy.png')

if __name__ == '__main__':
    plot_leapfrog(k_values, m, x0, v0, t)
    #plot_phase(s_values, k=2)
    #compare_energy(k_values, m, x0, v0, t)