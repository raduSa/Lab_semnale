import numpy as np
import matplotlib.pyplot as plt

for B in [1, 2, 4, 8]:
    x = np.linspace(-3, 3, 1000)
    func = lambda t: np.pow(np.sinc(B * t), 2)

    plt.plot(x, np.vectorize(func)(x))
    plt.title(f'Sinc, B={B}')
    plt.savefig(fname=f'Lab6/ex1_B={B}_fig1.pdf')
    plt.show()

    plt.figure(figsize=(16, 12))
    plt.title(f'B={B}')

    for i, fs in enumerate([1, 1.5, 2, 4]):
        Ts = 1 / fs
        samples = np.linspace(-3, 3, int(6 / Ts) + (int(6 / Ts)%2)^1)
        plt.subplot(4, 1, i+1)
        plt.plot(x, np.vectorize(func)(x))
        samples_eval = np.vectorize(func)(samples)
        plt.stem(samples, samples_eval)
        x_hat = lambda t: np.dot(samples_eval, np.sinc((t - samples) / Ts))
        print(samples, x_hat(0))
        plt.plot(x, np.vectorize(x_hat)(x), linestyle='dashed', color='orange')

    plt.tight_layout()    
    plt.savefig(fname=f'Lab6/ex1_B={B}_fig2.pdf')
    plt.show()