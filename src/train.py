import pandas as pd
from sklearn.model_selection import train_test_split
from model import BaselineModel

# Load data set
with open('data/train_set.csv', 'rb') as train_data:
    df = pd.read_csv(train_data, nrows=300)



df_train, df_test = train_test_split(df, test_size=0.2, random_state=62)


# Use this when iupred features are calculated and saved to file
#with open('data/iupred.csv', 'rb') as train_data:
#    df_iupred = pd.read_csv(iupred_features, nrows=100)
#df_iupred_train = df_iupred.iloc[df_train.index]
#df_iupred_test = df_iupred.iloc[df_test.index]


#print(df_train.index)

BaselineModel(model_file_path='src/model.pickle').train(df_train)#, extra_features=df_iupred_train)
