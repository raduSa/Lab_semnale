import numpy as np
import matplotlib.pyplot as plt

def convolve_4_times(x):
    for i in range(4):    
        plt.subplot(4, 1, i+1)
        plt.plot(np.linspace(0, 1, len(x)), x)

        x_new = list()
        for i in range(2 * len(x) - 1):
            new_elem = 0
            for k in range(len(x)):
                if 0 <= i - k < len(x):
                    new_elem += x[k] * x[i - k]
            x_new.append(new_elem)

        x = x_new

plt.figure(figsize=(16, 12))
plt.title('Random')
x = np.random.rand(100) * 10

convolve_4_times(x)

plt.savefig(fname=f'Lab6/ex2_fig1.pdf')
plt.show()


plt.figure(figsize=(16, 12))
plt.title('Block')
x = np.array([0]*25 + [1]*50 + [0]*25)

convolve_4_times(x)

plt.savefig(fname=f'Lab6/ex2_fig2.pdf')
plt.show()
