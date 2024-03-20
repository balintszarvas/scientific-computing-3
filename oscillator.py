import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

#contants
m = 2
k_values = [1]
x0 = 1
v0 = 0
t = np.linspace(0, 10, 1000)

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
    return x, v

def plot_leapfrog(k_values, m, x0, v0, t):
    figs = plt.subplots(len(k_values), 1)
    for k in k_values:
        x, v = leapfrog(k, m, x0, v0, t)
        plt.plot(t, x, label=f'x at k = {k}')
        plt.plot(t, v, label=f'v at k = {k}')
    
    plt.xlabel('Time')
    plt.ylabel('Position / Velocity')
    plt.title('Leapfrog Integration for Harmonic Oscillator')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase(k_values):
    fig = plt.subplots(3, 1)
    for i, k in enumerate(k_values):
        x, v = leapfrog(k, m, x0, v0, t)
        plt.subplot(3, 1, i+1)
        plt.plot(x, v)
        plt.title(f'k = {k}')
    plt.savefig('phase.png')

def runge_kutta(k, m, x0, v0, t):
    v = np.zeros(len(t))
    x = np.zeros(len(t))
    def f(t, y):
        x, v = y
        return [v, -k * x / m]

    sol = RK45(f, t[0], [x0, v0], t[-1])
    x[0] = x0
    v[0] = v0
    for i in range(1, len(t)):
        while sol.t < t[i]:
            sol.step()
        x[i], v[i] = sol.y

    return x, v
    
def calculate_energy(x, v, k, m):
    return 0.5 * m * v**2 + 0.5 * k * x**2

def compare_energy(k_values, m, x0, v0, t):
    figs = plt.subplots(len(k_values), 1)
    for method in ['leapfrog', 'runge_kutta']:
        for k in k_values:
            x, v = eval(method)(k, m, x0, v0, t)
            energy = calculate_energy(x, v, k, m)
            plt.plot(t, energy, label=f'{method} at k = {k}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.savefig('energy.png')


if __name__ == '__main__':
    compare_energy(k_values, m, x0, v0, t)