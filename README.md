# ASCM
This codebase contains the implementation for the paper [An asset subset-constrained minimax optimization framework for online portfolio selection](https://pdf.sciencedirectassets.com/271506/AIP/1-s2.0-S0957417424011655/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCbGkINXD9mrPGaZpmmeWmeBRF0c6wFaopTcsgHgCjCTgIgNKRtWQHh7UQvkgtIEb2qkBRhvqx1czuJxmAHLyOlrYgqswUIVRAFGgwwNTkwMDM1NDY4NjUiDLTsfuo1OdT%2BTalhMSqQBSO9kKqlYPNln2Xwc1D5q8KPg7y8CjLONuEssIX2tvkm630u4vtRxBEtL1brKRxarCcTLbTuYKm%2BjUNhHbAQVRqAI3eDzZAtaXf3tUZhzWyqa8vEZxHDhgR40JuLlxlVWNYVZZS8XUN63GsJKNNqE5prXuIB90jXLqLBHJxEFDmyDO5S%2B57FcMO7Ao7n8%2BO1WZ76q%2BsF9HhN0RlqXicVNIwbAaFeaBnnNZIUKdziM%2BmmnjdBP0zO%2Bm3aziK5imVQYSPvvqFXt1x7L5Lgbfppyvl2lJDKbhI16v0jeNU0ZFh0dkip6kuDeoa74THL9qKt84eveTp02UwdOAY1tGL%2BTXyF%2BYfSFkOlde7szhK%2BpR%2BtsVTkcvMt6eFHdqFrY5myQFS%2BZY%2B1Hmu2UTjtBbodvKydfVMctfH263JAQdK4c0Woj2G%2BCU87k1HvZOkG5pA59vbImuE6Z7FsCWFemXZKOCTAaIkznCHR%2FN4hYwr1aV5U0ft7Sc%2FABzAWlpTlnrRsV3q2K2ai3jvb7Em7fwhSgClT8LWNkSmbdUq9Ow4AQNTXKPgi04L0d0vQ4qHa0Y8ZGghlHfoCsk0H6Tocxhh0ov51KMJvni8uHR9NlcugFyYG7ym47P6EVnwOTNwmyc6d9mTUrJ0eu4WxGp69JA1FN%2Bb045nTE1Czhc5g3d%2BQ%2FmxIF5uW4A6fmGyTtyAxdfm5EgP%2BtczqTz9eoX87%2FKZ5KOyOBh6qAgJT9f1V4L3y7lSkwQ9SViZ44LXYcPtqe9Hh7P6GeOlwZHwSBt7HKWEI4VbY%2B6JpoM4xfW1SfPX%2FGHlPibOG%2F%2F0el0zwZjPz9%2B4PkqJAbIXiXU4mgd7r4ZXjI9tInjPhuuWgv5E8k86sbueiMPK26rIGOrEBPg1WNy8BuLmfJvvAyIYyZnPZZeeL6OTA7JOGHPr%2BicfdyKcV%2BVvPOTNPTpzO96OTEr3GXC37OhCpCCtEJhhsDj1E9eoYCQffarpAaMDypJwcf3jiUraDQRjXiFmiEf8zIoPUqdb7mb8g%2BtoxqN21lnvPOY24Pqnyzphs1bk8kc9%2BBmn9K%2FJ5YFREvba422RRir1OrpmKZV2wHjyXMLXNnsP1jY9daC2VppIvEM08pvbA&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240601T044602Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY26YPG4XX%2F20240601%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f949c56233638bba726c06a1f2a67b4c63edaf19cb9e3fa567264e0fd43aa075&hash=0b84e6117c0c5556c203189c0a3baf341f6ccdac24415922e9e1755fe520bd1c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0957417424011655&tid=spdf-d0c9d309-7598-4f77-a607-5ee05ed26028&sid=f662fe3a3ebd704b03092c342430ca6d4b83gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=07095e590104570a0d&rr=88cc8ec0cd10182a&cc=cn) (**ESWA**).

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

