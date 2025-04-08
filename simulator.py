import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_excel('century.xlsx', sheet_name = 'data')
vol = df['Volatility'].values[1:]
index = df['Price'].values
dividend = df['Dividends'].values
cpi = df['CPI'].values
N = len(vol)
lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
residVol = np.array([lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N - 1)])
nominal = np.array([np.log(index[k+1] + dividend[k+1]) - np.log(index[k]) for k in range(N)])
real = nominal - np.diff(np.log(cpi))
normReal = real/vol
normNominal = nominal/vol
NSIMS = 10000
NDISPLAYS = 5

def simReturns(infl, initialVol, nYears):
    if infl == 'R':
        means = np.mean(normReal)
        innovations = np.stack([residVol, normReal[1:]])
    if infl == 'N':
        means = np.mean(normNominal)
        innovations = np.stack([residVol, normNominal[1:]])
    innov = np.random.multivariate_normal([0, means], np.cov(innovations), nYears)
    simLVol = [np.log(initialVol)]
    simRet = []
    for t in range(nYears):
        simLVol.append(simLVol[-1]*betaVol + alphaVol + innov[t, 0])
        simRet.append(np.exp(simLVol[-1])*innov[t, 1])
    return simRet

def simWealth(infl, initialV, initialW, flow, horizon):
    returns = simReturns(infl, initialV, horizon)
    timeAvgRet = np.mean(returns)
    wealth = [initialW]
    for t in range(horizon):
        if (wealth[t] == 0) and (flow <= 0):
            wealth.append(0)
        else:
            new = max(wealth[t] * np.exp(returns[t]) + flow, 0)
            wealth.append(new)
    return timeAvgRet, np.array(wealth)

def output(infl, initialW, flow, horizon):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    timeAvgRets = []
    for sim in range(NSIMS):
        timeAvgRet, wealthSim = simWealth(infl, vol[-1], initialW, flow, horizon)
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
    simText = str(NSIMS) + ' Monte Carlo simulations'
    timeHorizonText = 'Time Horizon: ' + str(horizon) + ' years'
    if infl == 'N':
        inflText = 'Nominal returns, not inflation-adjusted'
    if infl == 'R':
        inflText = 'Real returns, inflation-adjusted'
    initWealthText = 'Initial Wealth ' + str(round(initialW))
    Portfolio = 'The portfolio: Standard & Poor 500 '
    SetupText = 'SETUP: ' + simText + '\n' + Portfolio + '\n' + timeHorizonText + '\n' + inflText + '\n' + initWealthText + '\n' + flowText + '\n'
    if np.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'average annual returns over paths without ruin ' + str(round(100*avgRet, 2)) + '%'
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
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 14})
    image_path = 'wealth.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    
output('R', 1000, -50, 40)