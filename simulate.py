import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Setup spatial grid and parameters
N = 500          # Number of spatial points
x = np.linspace(-50, 50, N)
dx = x[1] - x[0] # FIXED: Calculates the actual step size between points
dt = 0.01        # Time step size
total_frames = 200

# Physical parameters (Atomic units where hbar = 1, mass = 1)
x0 = -20.0       # Initial position center
sigma = 2.0      # Initial width of the packet
k0 = 5.0         # Initial momentum/velocity

# 2. Initialize the Gaussian Wave Function at t=0
psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx) # Normalization

# 3. Setup the Kinetic Energy operator in Fourier Space
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
kinetic_operator = np.exp(-1j * (k**2) * dt / 2)

# Setup the plotting environment
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-40, 40)
ax.set_ylim(0, 0.4)
ax.set_title("Time Evolution of a Quantum Gaussian Wave Packet", fontsize=12, fontweight='bold')
ax.set_xlabel("Position (x)", fontsize=10)
ax.set_ylabel("Probability Density |Ψ(x)|^2", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

line, = ax.plot([], [], lw=2.5, color='#1f77b4', label=r'$|\Psi(x,t)|^2$')
ax.legend(loc="upper right")

# 4. Simulation time-stepping function
def animate(frame):
    global psi
    # Step forward in time using Fast Fourier Transform (FFT)
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * kinetic_operator
    psi = np.fft.ifft(psi_k)
    
    # Calculate probability density
    prob_density = np.abs(psi)**2
    line.set_data(x, prob_density)
    return line,

# 5. Compile the animation loop and save as a web-ready GIF
print("Computing quantum wave packet states... Please wait.")
ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=30, blit=True)
ani.save('wave_packet.gif', writer='pillow', fps=30)
plt.close()
print("Success! 'wave_packet.gif' has been generated and downloaded.")
