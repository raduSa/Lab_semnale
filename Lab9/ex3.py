import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

trend_func = lambda x: x**2 + 5*x + 10
seasonal_func = lambda x: 5*np.sin(10*np.pi*x) + 3*np.cos(40*np.pi*x)
result_func = lambda x, y, z: x + y + z

N = 1000
x = np.linspace(0, 10, N)
trend = np.vectorize(trend_func)(x)
seasonal = np.vectorize(seasonal_func)(x)
noise = np.random.normal(0, 5, size=1000)
series = np.vectorize(result_func)(trend, seasonal, noise)

p = 2

def get_error(params, x, p):
    mu = params[1]
    theta = params[1:]
    N = len(x)
    epsilon = np.zeros(N)
    MSE = 0
    for i in range(N):        
        pred = mu
        for j in range(p):
            pred += theta[j] * (epsilon[i - 1 - j] if i - 1 - j >= 0 else 0)        

        pred = np.clip(pred, -1e6, 1e6)

        epsilon[i] = x[i] - pred
        MSE += np.pow(epsilon[i], 2)
    return MSE + cheap_penalty(theta)
# something something Add penalty to avoid non-invertible MA parameters
def cheap_penalty(theta):
    if np.any(np.abs(theta) > 0.99):
        return 1e12
    return 0

q = 2
mu = np.mean(series)

init = np.zeros(1 + q)
init[0] = mu
init[1:] = 0.01
print(init)
res = minimize(get_error, init, args=(x, q))
mu = res.x[0]
theta = res.x[1 : 1+q]

print(f'Got params: {mu, theta}')

epsilon = np.zeros(N)
predictions = np.zeros(N)
predictions[0] = mu
epsilon[0] = series[0] - predictions[0]

for i in range(1, N):
    prediction = mu
    for j in range(q):
        if i - 1 - j >= 0:
            prediction += theta[j] * epsilon[i - 1 - j]
    predictions[i] = prediction
    epsilon[i] = series[i] - predictions[i]

pred_manual = predictions

# try statsmodles MA model
p = 0
model = ARIMA(series, order=(p, 0, q))
fit = model.fit()
params_stats = fit.params

print(f'Got params (statsmodels): {params_stats}')

pred_stats = fit.predict()



plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(series, label='Original')
plt.plot(pred_manual, label=f'Manual MA({q})', color='green', linestyle='dotted')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(series, label='Original')
plt.plot(pred_stats, label=f'Statsmodels ARMA({p},{q})', color='green', linestyle='dotted')
plt.legend()

plt.tight_layout()
plt.savefig(fname=f'Lab9/ex3_fig1.pdf')
plt.show()