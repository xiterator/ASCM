
from universal.result import ListResult
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class MultiShower:

    def __init__(self, fileName):

        dtMark = str(datetime.datetime.now()) + '_'
        self.dataSet = fileName
        self.fileName = '/home/m/Desktop/new_result/' + fileName + '_' + dtMark + '.eps'

    def show(self, resultList, algoNameList, yLable='Total Wealth', logy1=True):

        res = ListResult(resultList,
                         algoNameList)
        d = res.to_dataframe()
        portfolio = d.copy()
        for name in portfolio.columns:

            ax = portfolio[name].plot(figsize=(7, 5), linewidth=1.5, logy=logy1)
        ax.legend()
        ax.set_ylabel(yLable)
        ax.set_xlabel('day')

        plt.show()
#
