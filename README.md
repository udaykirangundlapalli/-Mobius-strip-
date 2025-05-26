import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=100):
        self.R = R
        self.w = w
        self.n = n
        self.u, self.v = np.meshgrid(
            np.linspace(0, 2 * np.pi, n),
            np.linspace(-w / 2, w / 2, n)
        )
        self.x, self.y, self.z = self._compute_coordinates()

    def _compute_coordinates(self):
        u = self.u
        v = self.v
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        # Partial derivatives
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        xu = np.gradient(self.x, du, axis=0)
        xv = np.gradient(self.x, dv, axis=1)
        yu = np.gradient(self.y, du, axis=0)
        yv = np.gradient(self.y, dv, axis=1)
        zu = np.gradient(self.z, du, axis=0)
        zv = np.gradient(self.z, dv, axis=1)

        # Cross product of partials
        nx = yu * zv - zu * yv
        ny = zu * xv - xu * zv
        nz = xu * yv - yu * xv

        dA = np.sqrt(nx**2 + ny**2 + nz**2)
        area = simps(simps(dA, self.v[0]), self.u[:,0])
        return area

    def edge_length(self):
        edge_u = np.linspace(0, 2 * np.pi, self.n)
        edge_v = self.w / 2 * np.ones_like(edge_u)
        x_edge = (self.R + edge_v * np.cos(edge_u / 2)) * np.cos(edge_u)
        y_edge = (self.R + edge_v * np.cos(edge_u / 2)) * np.sin(edge_u)
        z_edge = edge_v * np.sin(edge_u / 2)

        dx = np.gradient(x_edge)
        dy = np.gradient(y_edge)
        dz = np.gradient(z_edge)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        length = simps(ds, edge_u)
        return length

    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, color='lightblue', edgecolor='gray', alpha=0.8)
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

# Create and visualize the Mobius strip
mobius = MobiusStrip(R=1.0, w=0.3, n=200)
area = mobius.surface_area()
length = mobius.edge_length()
mobius.plot()

area, length
