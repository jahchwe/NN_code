import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.cluster import KMeans



# define some nice functions to have for slopes and coordinates

def r2slope( r ):
    '''
    Take r coefficient, returns slope of one vector
    relative to another vector assumed to be a horizontal line
    '''
    deg = np.rad2deg(np.arccos(r))
    return np.tan(np.deg2rad(deg))

def rVecCoords( r ):
    '''
    Gets coordinates for vector with length 1 as slope for r
    relative to horizontal vector
    '''
    return [np.cos(np.arccos(r)),np.sin(np.arccos(r))]

# set r coefficient
r = .5

# create plot points
#### each array is a 2-point vector, [x1, y1, x2, y2]
soa = np.array([
    # coordinates for horizontal line (vector 1)
    [0, 0, 1, 0],
    [0, 0, -1, 0],
    # coordinates for related vector, vector 2
    [0, 0, rVecCoords(r)[0], rVecCoords(r)[1]],
    [0, 0, -1*rVecCoords(r)[0], -1*rVecCoords(r)[1]]
])

# plot
X, Y, U, V = zip(*soa)
plt.figure()
ax = plt.gca()
ax.annotate("", xy=(0, 1), xytext=(0, -1), arrowprops=dict(arrowstyle="<->"), color='gray')
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_aspect('equal')
plt.axis('off')
plt.draw()
plt.show()
