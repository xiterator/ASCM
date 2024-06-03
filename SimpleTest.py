
from universal import tools
from universal import algos
import logging

# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import matplotlib.pyplot as plt

#print('type: ', type(matplotlib.rcParams['savefig.dpi']), 'va: ', matplotlib.rcParams['savefig.dpi'])

from MultiShower import MultiShower


# increase the size of graphs

class Tester:

    def __init__(self):
        self.data = None
        self.algo = None
        self.result = None
        self.X = None
        self.datasetName = None
        self.NStocks = 0

    def createDataSet(self, datasetName):
        # load data using tools module
        self.data = tools.dataset(datasetName)
        print(self.data)
        self.datasetName = datasetName
        print('data.type: ', type(self.data))

        self.NStocks = self.data.shape[1]
        print(self.data.head())
        print(self.data.shape)

    def createAlgo(self, fileName):

        self.algo = algos.OLMAR()

        return self.algo

    def runAlgo(self):
        self.result = self.algo.run(self.data)

    def showResult(self, d):

        from universal.algos.ascm import ASCM
        from universal.algos.bah import BAH


        result_ascm = ASCM( rho_l_alpha=0.15, rho_h_alpha=0.45, rho_l_beta=0.654, rho_h_beta=0.734, subsets_comb=31, lam=0.002, w_alpha=9, w_beta=5).run(self.data)
        result_bah = BAH().run(self.data)

        ms = MultiShower(self.datasetName)

        for fee in [0]:
            result_bah.fee = fee
            result_ascm.fee = fee

            ms.show(
                [
                    result_bah,
                    result_ascm,
                ],
                [
                    'BAH',
                    'ASCM',

                ],
                yLable=self.datasetName + ' Cumulative Wealth')

            plt.show()
            print('ASCM results ' + result_ascm.summary())

    @staticmethod
    def testSimple():

        datasets = ['djia','hs300','crypto','sp500','tse','EuroStoxx50']
        for d in datasets:
            t = Tester()
            t.createDataSet(d)
            # t.createAlgo(d)
            # t.runAlgo()
            t.showResult(d)


if __name__ == '__main__':
    Tester.testSimple()

