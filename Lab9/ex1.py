import numpy as np
import matplotlib.pyplot as plt

trend_func = lambda x: x**2 + 5*x + 10
seasonal_func = lambda x: 5*np.sin(10*np.pi*x) + 3*np.cos(40*np.pi*x)
result_func = lambda x, y, z: x + y + z

N = 1000
x = np.linspace(0, 10, N)
trend = np.vectorize(trend_func)(x)
seasonal = np.vectorize(seasonal_func)(x)
noise = np.random.normal(0, 5, size=1000)
series = np.vectorize(result_func)(trend, seasonal, noise)

plt.subplot(4, 1, 1)
plt.title('Trend')
plt.plot(x, trend)

plt.subplot(4, 1, 2)
plt.title('Seasonal')
plt.plot(x, seasonal)

plt.subplot(4, 1, 3)
plt.title('Noise')
plt.plot(x, noise)

plt.subplot(4, 1, 4)
plt.title('Series')
plt.plot(x, series)

plt.tight_layout()
plt.savefig(fname=f'Lab9/ex1_fig1.pdf')
plt.show()

# Exp mean

# Find best alpha
search = np.linspace(0, 1, 100)[1:]
min_error = float('inf')
best_alpha = None
best_predictions = None
for alpha in search:
    s = np.zeros(N)
    s[0] = series[0]

    error = np.pow(s[0] - series[1], 2)

    for i in range(1, N):
        s[i] = alpha * series[i] + (1 - alpha) * s[i - 1]
        if i < N - 1:
            error += np.pow(s[i] - series[i+1], 2)
    
    plt.scatter(alpha, error)
    
    if error < min_error:
        min_error = error
        best_alpha = alpha  
        best_predictions = s

plt.title(f'Alpha search - best = {best_alpha}')
plt.savefig(fname=f'Lab9/ex1_fig3.pdf')
plt.show()  

# plot prediction

plt.subplot(2, 1, 1)
plt.title(f'Exp mean for alpha = {best_alpha}')
plt.plot(x, best_predictions)

plt.subplot(2, 1, 2)
plt.plot(x, series)
plt.title('Original')
plt.savefig(fname=f'Lab9/ex1_fig2.pdf')
plt.show()



# Double

# Find best alpha, beta
min_error = float('inf')
best_alpha = None
best_beta = None
best_predictions = None
for alpha in [0, 1, 0.05]:
    for beta in [0, 1, 0.05]:
        s = np.zeros(N)
        s[0] = series[0]

        b = np.zeros(N)
        b[0] = series[1] - series[0]

        predicted = np.zeros(N)

        error = np.pow(s[0] - series[1], 2)

        for i in range(1, N):
            s[i] = alpha * series[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
            b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
            predicted[i] = s[i] + b[i]
            if i < N - 1:
                error += np.pow(predicted[i] - series[i+1], 2)            
        
        if error < min_error:
            min_error = error
            best_alpha = alpha  
            best_beta = beta
            best_predictions = predicted

# plot prediction

plt.subplot(2, 1, 1)
plt.title(f'Double exp mean for alpha = {best_alpha}, beta = {best_beta}')
plt.plot(x, best_predictions)

plt.subplot(2, 1, 2)
plt.plot(x, series)
plt.title('Original')
plt.savefig(fname=f'Lab9/ex1_fig4.pdf')
plt.show()



# Triple 
s = np.zeros(N)
s[0] = series[0]

alpha = 0.5

b = np.zeros(N)
b[0] = series[1] - series[0]

beta = 0.5

L = 200
c = np.zeros(N)
c[:L] = [series[i] - series[0] for i in range(L)]

gamma = 0.5

predicted = np.zeros(N)
for i in range(1, N):
    c_prev = c[i - L] if i - L >= 0 else 0
    s[i] = alpha * (series[i] - c_prev) + (1 - alpha) * (s[i - 1] + b[i - 1])
    b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    c[i] = gamma * (series[i] - s[i] - b[i - 1]) + (1 - gamma) * c_prev
    if i - L + 1 >= 0:
        predicted[i] = s[i] + b[i] + c[i - L + 1]

plt.subplot(2, 1, 1)
plt.title(f'Triple exp mean for alpha = {alpha}, beta = {beta}, gamma = {gamma}')
plt.plot(x, predicted)

plt.subplot(2, 1, 2)
plt.plot(x, series)
plt.title('Original')
plt.savefig(fname=f'Lab9/ex1_fig5.pdf')
plt.show()
