---
title: "Time Series Forecasting using ARIMA"
date: 2024-09-21
tags: [Machine Learning]
mathjax: true
mathjaxEnableSingleDollar: true
---


ARIMA is a well-known model for time series forecasting. In this post we will go over the building blocks: MA(q) models, AR(p) models and ARMA(p,q) models of such a powerful model.
<!--more-->

ARIMA(p, q, d) model is a combination of:

- I(d): integration (reverse of differencing) component
- AR(p): auto regressive component
- MA(q): moving average component

# Integration: I(d)

The moving average (MA) and auto regressive (AR) models, requires the time series to be stationary. A time series is tested for stationarity using the [Augmented Dickey–Fuller test(ADF)](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) test and a non-stationary time series data is made stationary using differencing. We are interested in the order of integration  $d$ which is the **minimum** number of times a series must be differenced to become stationary. The differenced time series $y^{(d)}$ is then used in subsequent models.

# Auto Regressive: AR(p)

The auto regressive (AR) model is a regression of a variable against itself i.e, the current value is a **linear** combination of the past **values**. The current value is represented as:

$$
y_t = C + \epsilon_t + \phi_1y_{t-1} +_ ... \phi_py_{t-p}
$$

Where:
- $C$ is a constant term, 
- $\epsilon_t$ is the current error term $i$,
- $\phi_i$ is the coefficient of the series value at time $i$ 
- $p$ is the order of the AR model.


# Moving Average: MA(q)

The moving average (MA) model, states that the current value is a **linear** combination on the current and past **error** terms. The current value is represented as:

$$
y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} +_ ... \theta_q\epsilon_{t-q}
$$

Where:
- $\mu$ is the series average, 
- $\epsilon_i$ are the error terms at time $i$,
- $\theta_i$ is the coefficient of the error term at time $i$ 
- $q$ is the order of the MA model.


# Auto Regressive Moving Average: ARMA(p, q)

The auto regressive moving average (ARMA) model is a combination of the moving average (MR) model and the auto regressive (AR).

$$
y_t = C + \phi_1y_{t-1} + ... \phi_py_{t-p} + \mu + \theta_1\epsilon_{t-1} + ... \theta_q\epsilon_{t-q}  + \epsilon_t 
$$

Where:
- $C$ is a constant term, 
- $\mu$ is the series average, 
- $\epsilon_i$ are the error terms at time $i$,
- $q$ is the order of the MA model.
- $\theta_i$ is the coefficient of the error term at time $i$ 
- $p$ is the order of the AR model.
- $\phi_i$ is the coefficient of the series value at time $i$ 


# Putting it all together

The auto regressive integrated moving average (ARIMA) model is the same as the ARIMA model along with the integration component i.e., it is a combination of the moving average (MR) model, the auto regressive (AR) model using the differenced series.
 
$$
y_{t}^{(d)} = C + \phi_1y_{t-1}^{(d)} + ... \phi_py_{t-p}^{(d)} + \mu + \theta_1\epsilon_{t-1}^{(d)} + ... \theta_q\epsilon_{t-q}^{(d)}  + \epsilon_t 
$$

Fortunately, in practice we don't have to implement these models from scratch as [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) implementation is available in `statsmodel`. They also provide implementation for more advanced models such as [SARIMA and SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX) that can handle seasonality and external features. I recommend [Time Series Forecasting in Python book](https://www.manning.com/books/time-series-forecasting-in-python-book) to dive deeper into the fascinating world of time series forecasting.