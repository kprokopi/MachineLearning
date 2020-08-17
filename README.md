# Machine Learning
(Machine Learning in Python)

### Instacart Market Basket Analysis
https://www.kaggle.com/c/instacart-market-basket-analysis

Instacart is an American company that operates as a same-day grocery delivery service. Customers select groceries through a web application from various retailers and delivered by a personal shopper. Instacart's service is mainly provided through a smartphone app, available on iOS and Android platforms, apart from its website.

In 2017 Instacart organised a Kaggle competition and provided to the community a sample dataset of over 3 million grocery orders from more than 200,000 Instacart users. The orders include 32 million basket items and 50,000 unique products. The objective of the competition was participants to predict which previously purchased products will be in a user’s next order.

The notebooks can be run in [colab](https://colab.research.google.com) 

All codes use [XGboost algorithm](https://xgboost.readthedocs.io/en/latest/#).

Python notebooks available:
1. with 12 features,
2. with 17 features
3. with 18 features (experimental)


Python codes available:
1. with 17 features,
2. with 17 features with GPU support
3. with 17 with GridSearchCV
4. with 18 features (experimental)

The code **fetch_csv.py** fetch the csv files from kaggle and the code **submit_csv.py** submit the csv file to kaggle.


