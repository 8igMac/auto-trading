from enum import IntEnum
import pandas as pd

class Action(IntEnum):
    BUY = 1
    HOLD = 0
    SELL = -1

def load_data(csv_file: str) -> pd.DataFrame:
    COLUMN_NAMES = ('open', 'high', 'low', 'close')
    stock_df = pd.read_csv(csv_file, names=COLUMN_NAMES)
    return stock_df

class Trader:

    def _buy_and_hold(self, row: pd.core.series.Series, i: int) -> Action:
        if i == 0:
            return Action.BUY
        else:
            return Action.HOLD

    def predict_action(self, row: pd.core.series.Series, i: int) -> Action:
        # Access the row like: row['open'], row['high'], row['low'], row['close'].
        action = self._buy_and_hold(row, i)
        return action

    def train(self, training_data: pd.DataFrame):
        print('training')
        pass

    def re_training(self, row: pd.core.series.Series, i: int):
        print('re_training')
        pass

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
