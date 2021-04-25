import pandas as pd
from sklearn.model_selection import train_test_split
from model import BaselineModel

nrows = 10000

# Load data set
with open('data/train_set.csv', 'rb') as train_data:
    df = pd.read_csv(train_data, nrows=nrows)


df_train, df_test = train_test_split(df, test_size=0.2, random_state=62)


# Use this when iupred features are calculated and saved to file3
with open('data/iupred.csv', 'rb') as iupred_features:
    df_iupred = pd.read_csv(iupred_features, nrows=nrows)
df_iupred_train = df_iupred.iloc[df_train.index]
df_iupred_test = df_iupred.iloc[df_test.index]


# mode can be 'TRAIN', 'TUNE' or 'TRAIN_AND_TUNE'
# search_mode can be 'RANDOM_SEARCH' or 'GRID_SEARCH'
# extra_features is optional
# model_type can be 'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor'
BaselineModel(model_file_path='src/model.pickle').train(df_train, extra_features=df_iupred_train,
                                                        mode='TUNE_AND_TRAIN',
                                                        search_mode='GRID_SEARCH',
                                                        model_type='DecisionTreeRegressor',
                                                        verbose=1)
