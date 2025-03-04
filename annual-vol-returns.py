import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', stats.skew(data))
    print('Kurtosis:', stats.kurtosis(data))
    print('Shapiro-Wilk p = ', stats.shapiro(data)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(data)[1])
    
df = pd.read_excel('annual-vol-returns.xlsx', sheet_name = 'data')
vol = df['Volatility'].values
N = len(vol) - 1
lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
print('Slope = ', round(betaVol, 3))
print('Intercept = ', round(alphaVol, 3))
residVol = np.array([lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N)])
real = df['Real'].values
nominal = df['Nominal'].values
normReal = real/vol
normNominal = nominal/vol
meanReal = np.mean(normReal)
meanNominal = np.mean(normNominal)
print('Mean values for normalized real returns = ', round(meanReal, 4))
print('Mean values for normalized nominal returns = ', round(meanNominal, 4))
covReal = np.cov(np.stack([residVol, normReal[1:]]))
print('covariance for real version')
print(covReal)
covNominal = np.cov(np.stack([residVol, normNominal[1:]]))
print('covariance for nominal version')
print(covNominal)

plots(residVol, 'AR(1) Volatility Residuals')
plots(normReal, 'Normalized Real Returns')
plots(normNominal, 'Normalized Nominal Returns')
analysis(residVol, 'AR(1) Volatility Residuals')
analysis(normReal, 'Normalized Real Returns')
analysis(normNominal, 'Normalized Nominal Returns')

NSIMS = 1000
NDISPLAYS = 5
np.random.seed(0)

def simReturns(infl, initialVol, nYears):
    if infl == 'R':
        mean = meanReal
        cov = covReal
    if infl == 'N':
        mean = meanNominal
        cov = covNominal
    innov = np.random.multivariate_normal([0, 0], cov, nYears)
    simLVol = [np.log(initialVol)]
    simRet = []
    for t in range(nYears):
        simRet.append(np.exp(simLVol[-1])*(innov[t, 1] + mean))
        simLVol.append(simLVol[-1]*betaVol + alphaVol + innov[t, 0])
    return simRet

def simWealth(infl, initialV, initialW, flow, horizon):
    returns = simReturns(infl, initialV, horizon)
    timeAvgRet = np.mean(returns)
    wealth = [initialW]
    for t in range(horizon):
        if (wealth[t] == 0):
            wealth.append(0)
        else:
            new = max(wealth[t] * np.exp(returns[t]) + flow, 0)
            wealth.append(new)
    return timeAvgRet, np.array(wealth)

def output(infl, initialV, initialW, flow, horizon):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    timeAvgRets = []
    for sim in range(NSIMS):
        timeAvgRet, wealthSim = simWealth(infl, initialV, initialW, flow, horizon)
        timeAvgRets.append(timeAvgRet)
        paths.append(wealthSim)
    paths = np.array(paths)
    avgRet = np.mean([timeAvgRets[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = np.mean(paths[:, -1])
    meanProb = np.mean([paths[sim, -1] > wealthMean for sim in range(NSIMS)])
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])
    sortedIndices = np.argsort(paths[:, -1])
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]
    times = range(horizon + 1)
    simText = str(NSIMS) + ' of simulations'
    timeHorizonText = 'Time Horizon: ' + str(horizon) + ' years'
    if infl == 'N':
        inflText = 'Nominal returns, not inflation-adjusted'
    if infl == 'R':
        inflText = 'Real returns, inflation-adjusted'
    initWealthText = 'Initial Wealth ' + str(round(initialW))
    Portfolio = 'The portfolio: 100% Large Stocks'
    initMarketText = 'Initial conditions: Volatility ' + str(round(initialV, 2)) 
    SetupText = 'SETUP: ' + simText + '\n' + Portfolio + '\n' + timeHorizonText + '\n' + inflText + '\n' + initWealthText +'\n' + initMarketText + '\n' + flowText + '\n'
    if np.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'time averaged annual returns:\naverage over all paths without ruin ' + str(round(100*avgRet, 2)) + '%'
        MeanText = 'average final wealth ' + str(round(wealthMean))
        MeanCompText = 'final wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + MeanCompText
    bigTitle = SetupText + '\n' + ResultText + '\n'
    plt.plot([0], [initialW], color = 'w', label = bigTitle)
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth, ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            plt.plot(times, paths[index], label = '0' + rankText + 'Gone Bust !!!')
        else:
            plt.plot(times, paths[index], label = str(selectTerminalWealth) + rankText + 'returns: ' + str(round(100*timeAvgRets[index], 2)) + '%')
    plt.xlabel('Years')
    plt.ylabel('Wealth')
    plt.title('Wealth Plot')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 12})
    image_path = 'wealth.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    
output('R', 11.7, 1000, -75, 20)