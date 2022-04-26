# auto-trading
An auto stock trading bot.

## Setup
```sh
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Usage
```sh
$ python trader.py --training data/training.csv --testing data/testing.csv --output output.csv
```

## Idea
Buy on the lowest point. Sell on the highest point. 
Use yesterday and the day before yesterday to see if we are on the 
lowest/highest point.
