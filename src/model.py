from sklearn.svm import SVR
import numpy as np
import pickle


class BaselineModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def vectorize_sequences(self, sequence_array):
        vectorize_on_length = np.vectorize(len)
        return np.reshape(vectorize_on_length(sequence_array), (-1, 1))

    def train(self, df_train):
        X = self.vectorize_sequences(df_train['sequence'].to_numpy())
        y = df_train['mean_growth_PH'].to_numpy()

        model = SVR()
        model.fit(X, y)

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        return model.predict(X_vectorized)
