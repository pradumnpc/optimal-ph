import argparse
import pandas as pd
from model import BaselineModel

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', default='input.csv')
args = parser.parse_args()

# Config
output_file_path = 'predictions.csv'

# Load input.csv
with open(args.input_csv) as input_csv:
    df = pd.read_csv(input_csv)
# Load iupred.csv
with open('submission/iupred.csv', 'rb') as iupred_features:
    df_iupred = pd.read_csv(iupred_features)

# Run predictions
y_predictions = BaselineModel(model_file_path='src/model.pickle').predict(df,extra_features=df_iupred)


# Save predictions to file
df_predictions = pd.DataFrame({'prediction': y_predictions})
df_predictions.to_csv(output_file_path, index=False)

print(f'{len(y_predictions)} predictions saved to a csv file')
