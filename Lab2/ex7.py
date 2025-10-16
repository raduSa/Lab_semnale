import numpy as np
import matplotlib.pyplot as plt

signal = lambda t : np.sin(4*np.pi * t)
x = np.linspace(0, 1, 1000)

fig, axs = plt.subplots(3)
y = np.vectorize(signal)(x)
axs[0].stem(x, y)

x_decimated = x[0::4]
y = np.vectorize(signal)(x_decimated)
axs[1].stem(x_decimated, y)

x_decimated_2 = x_decimated[0::4]
y = np.vectorize(signal)(x_decimated_2)
axs[2].stem(x_decimated_2, y)

plt.show()