# ASCM
This codebase contains the implementation for the paper [An asset subset-constrained minimax optimization framework for online portfolio selection](https://doi.org/10.1016/j.eswa.2024.124299) (**ESWA**).

In this paper, we propose an asset subset-constrained minimax (ASCM) optimization framework, which generates optimal portfolios from diverse investment strategies represented as asset subsets.

## Installation

#### Prerequisites

- **Operating System**: tested on Window 10.
- **Python Version**: 3.8.0.

#### Dependencies

You can install the dependencies by:

```
   pip install -r requirements.txt
```

## Details

#### Datasets
We tested the algorithms performance on six real datasets, namely djia, hs300, crypto, sp500, tse, and EuroStoxx50 which stored in `universal\data` directory. The first row in the dataset represents the stock name, and each subsequent row represents the closing price of all stocks for the same trading period. If you want to test other datasets, you can first store the dataset in pkl format in the `data` directory, and then modify the list of test datasets in the `testSimple()` function under `SimpleTest.py`.

#### Comparison algorithm

This project includes two tested algorithms, namely ASCM and benchmark algorithm BAH(Buy ang Hold), which are stored in `universal\algos` directory. If you want to add other comparison algorithms, you can emulate the ascm algorithm, rewrite the `step()` function, store the [algorithm name].py file in the `universal\algos` directory, import the algorithm package in `__init__.py` in that directory, and finally add the algorithm running code in the `showResult()` function in the `SimpleTest.py`.

#### Hyperparameters

The hyperparameters of each algorithm and transaction cost can be modified according to your needs in the function `showResult()` under `SimpleTest.py`. For instance, there is an algorithm instance `result_ascm`, and you can modify its transaction cost to 1‰:
```
   result_ascm.fee = 0.001
```

## Running

You can run the code using the following command:

```
   python SimpleTest.py
```
By default, the program will output the prices of the current test dataset, the specific progress of each algorithm, various performance indicators (cumulative wealth, sharpe ratio, max drawdown, etc.) of ASCM, and an EPS format image which will be stored in `experimentFigure\cumulative wealth\[dataset name].eps`.


## Citation
If you find our paper or code is useful, please consider citing:
```bash
   @article{YIN2024124299,
    title = {An asset subset-constrained minimax optimization framework for online portfolio selection},
    journal = {Expert Systems with Applications},
    pages = {124299},
    year = {2024},
    issn = {0957-4174},
    author = {Jianfei Yin and Anyang Zhong and Xiaomian Xiao and Ruili Wang and Joshua Zhexue Huang},
}
```

