import numpy as np
import matplotlib.pyplot as plt
import random_gradient_generator as rgg

from scipy import interpolate

# Generate function
x, y, z = rgg.generate_random_function(-3, 3, 0.1)

# Downsample
sample_rate = 1
x_down = x[::sample_rate]
y_down = y[::sample_rate]
z_down = z[::sample_rate]


r = np.stack([x, y]).T

# This works incredibly well for gradient propagation (at least for smooth functions)
# S = interpolate.CloughTocher2DInterpolator(r, z)
# S = interpolate.LinearNDInterpolator(r, z)
S = interpolate.Rbf(x_down, y_down, z_down, epsilon=0.1, function='thin_plate')

N = 400
xl = np.linspace(x_down.min(), x_down.max(), N)
yl = np.linspace(y_down.min(), y_down.max(), N)
X, Y = np.meshgrid(xl, yl)

Zp = S(X.ravel(), Y.ravel())
Z = Zp.reshape(X.shape)

dZdy, dZdx = np.gradient(Z, yl, xl, edge_order=1)

SdZx = np.nancumsum(dZdx, axis=1)*np.diff(xl)[0]
SdZy = np.nancumsum(dZdy, axis=0)*np.diff(yl)[0]

Zhat = np.zeros(SdZx.shape)
for i in range(Zhat.shape[0]):
    for j in range(Zhat.shape[1]):
        Zhat[i,j] += np.nansum([SdZx[0,N//2], SdZy[i,N//2], SdZx[i,j], -SdZx[i,N//2]])

z_trans = [30, 60]       
Zhat += Z[100,100] - Zhat[100,100] + z_trans[1]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, z+z_trans[0], linewidth=0, alpha=0.5, color='blue')
surf_1 = ax.plot_surface(x_down, y_down, z_down, linewidth=0, alpha=1.0, color='lime')
reconstructed_surf = ax.plot_surface(X, Y, Zhat, linewidth=0, alpha=0.8, color='blueviolet')
plt.show()