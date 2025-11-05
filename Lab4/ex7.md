## Ps_dB = 90dB, SNR_dB = 80dB


$$
\mathrm{SNR_{dB}} = 10 \log_{10}(\mathrm{SNR}) = 10 \log_{10}\!\left(\frac{P_s}{P_n}\right) \tag{1}
$$

$$
\mathrm{P_{s,dB}} = 10 \log_{10}\!\left(\frac{P_s}{P_0}\right) 
\;\Rightarrow\;
P_s = 10^{\frac{\mathrm{P_{s,dB}}}{10}} P_0 
\tag{2}
$$

$$
\mathrm{P_{n,dB}} = 10 \log_{10}\!\left(\frac{P_n}{P_0}\right) 
\;\Rightarrow\;
P_n = 10^{\frac{\mathrm{P_{n,dB}}}{10}} P_0 
\tag{3}
$$

$$
(1), (2), (3) \;\Rightarrow\;
\mathrm{SNR_{dB}} = 10 \log_{10}\!\left(10^{\frac{\mathrm{P_{s,dB}}}{10} - \frac{\mathrm{P_{n,dB}}}{10}}\right) 
= \mathrm{P_{s,dB}} - \mathrm{P_{n,dB}}
$$

$$
\mathrm{P_{n,dB}} = \mathrm{P_{s,dB}} - \mathrm{SNR_{dB}} = 10\,\mathrm{dB}
$$
