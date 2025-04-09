# annual-simulator
Data and code for annualized version of the financial simulator, with only total returns and volatility. Log volatility is modeled as an autoregression of order 1. Total (real or nominal) returns after dividing by volatility is modeled by Gaussian IID. 

We analyze residuals for this autoregression and these normalized returns using autocorrelation function for white noise and the quantile-quantile plot for normality. The data is taken from my web page. 

Update: I corrected some mistakes, and afapted the code simulator.py to use the same data file which is used in further research: century.xlsx. It has raw annual data on the S&P index level (end of year), dividends, CPI (December), and volatility. See the blog post https://my-finance.org/2025/04/08/updated-annual-simulator-online-live/
