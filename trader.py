from enum import IntEnum
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
import numpy as np
import time 

class Action(IntEnum):
    BUY = 1
    HOLD = 0
    SELL = -1

def load_data(csv_file: str) -> pd.DataFrame:
    COLUMN_NAMES = ('open', 'high', 'low', 'close')
    stock_df = pd.read_csv(csv_file, names=COLUMN_NAMES)
    return stock_df

class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    super(LSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim,  output_dim)
  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
    out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
    out = self.fc(out[:, -1, :])
    return out

class Trader:
    def __init__(self):
        self.stock = 0
        self.bought = False
        self.hist = list()
        self.look_back = 51

    def _buy_and_hold(self, row: pd.core.series.Series, i: int) -> Action:
        if i == 0:
            return Action.BUY
        else:
            return Action.HOLD

    def _slower(self, row: pd.core.series.Series, i: int) -> Action:
        yesterday = self.training_df.iloc[-1]
        before_yesterday = self.training_df.iloc[-2]

        if i == 1:
            self.init_open = row['open']

        if i == 0:
            self.stock += 1
            return Action.BUY
        elif yesterday['open'] > before_yesterday['open'] and yesterday['open'] > row['open'] and row['open'] > self.init_open and self.stock > -1:
            # Local maxima.
            self.stock -= 1
            return Action.SELL
        elif yesterday['open'] < before_yesterday['open'] and yesterday['open'] < row['open'] and row['open'] < self.init_open and self.stock < 1:
            # Local minima.
            self.stock += 1
            return Action.BUY
        else:
            return Action.HOLD

    def _prepare_predict_data(self, open_price: float):
        # Normalize data.
        price = self.training_df[['open']]
        price = price.append({'open': open_price}, ignore_index=True)
        price_np = price['open'].values
        scaler = StandardScaler()
        price['open'] = scaler.fit_transform(price_np.reshape(-1, 1))

        # window.
        data_raw = price.to_numpy()
        data = []

        # Shift the data.
        for index in range(len(data_raw) - lookback):
          data.append(data_raw[index:index + lookback])
        
        data = np.array(data)

        x_train = data[:, :-1, :]
        y_train = data[:, -1, :]
        data_raw = price.to_numpy()
        return np.array([data_raw[-1 * (self.look_back - 1):]])

    def _lstm(self, row: pd.core.series.Series, i: int) -> Action:

        # Record second day open price.
        if i == 1:
            self.second_day_open = row['open']

        # LSTM Predict.
        with torch.no_grad():
            x_test = self._prepare_predict_data(row['open'])
            self.hist.append(self.model(x_test))
        print(self.hist) # debug

        # Calculate slope.
        slope = self.hist[-1] - self.hist[-2]
            
        if slope < 0 and row['open'] < self.second_day_open and not self.bought:
            self.bought = True
            return Action.BUY
        else:
            return Action.HOLD
        
    def predict_action(self, row: pd.core.series.Series, i: int) -> Action:
        # Access the row like: row['open'], row['high'], row['low'], row['close'].
        # action = self._lstm(row, i)
        action = self._slower(row, i)
        return action

    def _split_data(self, stock, lookback):
      data_raw = stock.to_numpy()
      data = []

      # Shift the data.
      for index in range(len(data_raw) - lookback):
        data.append(data_raw[index:index + lookback])
      
      data = np.array(data)

      x_train = data[:, :-1, :]
      y_train = data[:, -1, :]

      return x_train, y_train

    def train(self, training_data_df: pd.DataFrame):
        # Record the training data.
        self.training_df = training_data_df
        
        # Normalize data.
        price = training_data_df[['open']]
        scaler = StandardScaler()
        price['open'] = scaler.fit_transform(price['open'].values.reshape(-1, 1))

        # Split the data
        x_train, y_train = self._split_data(price, self.look_back)

        # Conver numpy to tensor.
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)

        # Define the model.
        input_dim = 1
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        num_epochs = 5 # TODO: increase the epoch.

        self.model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Train the model.
        hist = np.zeros(num_epochs)
        start_time = time.time()
        for t in range(num_epochs):
          y_pred = self.model(x_train)

          loss = criterion(y_pred, y_train)
          print(f'Epoch: {t}, MSE: {loss.item()}')
          hist[t] = loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        training_time = time.time() - start_time
        print(f'training time: {training_time}')

    def re_training(self, row: pd.core.series.Series, i: int):
        # TODO: prettier way to append dataframe.
        d = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
        }
        df = pd.DataFrame(data=d, index=[0])
        self.training_df = pd.concat([self.training_df, df], ignore_index=True)

if __name__ == '__main__':
    # You should not modify this part.
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # Train the trader.
    training_data_df = load_data(args.training)
    trader = Trader()
    trader.train(training_data_df)
    
    # Trader perdict actions.
    testing_data_df = load_data(args.testing)
    with open(args.output, 'w') as output_file:
        # Drop the last row cause we don't predict on the last day.
        testing_data_df.drop(len(testing_data_df)-1, inplace=True)

        for i, row in testing_data_df.iterrows():
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row, i)
            output_file.write(f'{action.value}\n')

            # Retraining on the new data.
            trader.re_training(row, i)
