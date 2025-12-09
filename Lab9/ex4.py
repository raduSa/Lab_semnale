import numpy as np
import matplotlib.pyplot as plt
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


best_params = (0, 0)
best_model = None
best_aic = np.inf
# try 5 vals for p and q    
for p in range(1, 22, 5):
    for q in range(1, 22, 5):
        model = ARIMA(series, order=(p, 0, q))
        fit = model.fit()
        
        if fit.aic < best_aic:
            best_params = (p, q)
            best_model = fit
            best_aic = fit.aic

# again
for p in range(max(0, best_params[0] - 2), min(21, best_params[0] + 2)):
    for q in range(max(0, best_params[1] - 2), min(21, best_params[1] + 2)):
        model = ARIMA(series, order=(p, 0, q))
        fit = model.fit()
        
        if fit.aic < best_aic:
            best_params = (p, q)
            best_model = fit
            best_aic = fit.aic

plt.figure(figsize=(14, 8))

plt.plot(series, label='Original')
plt.plot(best_model.predict(), label=f'ARMA({p},{q})', color='green', linestyle='dotted')
plt.legend()

plt.tight_layout()
plt.savefig(fname=f'Lab9/ex4_fig1.pdf')
plt.show()