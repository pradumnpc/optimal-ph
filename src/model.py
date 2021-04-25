from sklearn import tree
import numpy as np
import pickle
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

class BaselineModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
    
    def ss_features(self, seq):
        features = np.array([len(seq),3,5])
        return(features)
        
    def seq_features(self, seq):
        aacode_1 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        seq = seq.replace('X','A')
        features = []
        
        analysed_seq = ProteinAnalysis(seq)
        features.append(analysed_seq.molecular_weight())
        features.append(analysed_seq.aromaticity())
        features.append(analysed_seq.instability_index())
        features.append(analysed_seq.gravy())
        features.append(analysed_seq.isoelectric_point())
        features.append(analysed_seq.charge_at_pH(6.0))
        features.append(analysed_seq.charge_at_pH(7.0))
        features.append(analysed_seq.charge_at_pH(8.0))
        features += list(analysed_seq.secondary_structure_fraction())
        aapercent = analysed_seq.get_amino_acids_percent()
        features += [aapercent[aa] for aa in aacode_1]
        return(np.array(features))


    def seq_cat6_features(self, seq):
        l = float(len(seq))
        
        return(np.array([(seq.count('H')+seq.count('R')+seq.count('K')) / l,
                         (seq.count('D')+seq.count('E')+seq.count('N')+seq.count('Q')) / l,
                         (seq.count('C')) / l,
                         (seq.count('S')+seq.count('T')+seq.count('P')+seq.count('A')+seq.count('G')) / l,
                         (seq.count('M')+seq.count('I')+seq.count('L')+seq.count('V')) / l,
                         (seq.count('F')+seq.count('Y')+seq.count('W')) / l ]))
    
    def get_nterm_seq(self, seq):
        num_aa = 10
        return(seq[:num_aa])
    
    def get_cterm_seq(self, seq):
        num_aa = 10
        return(seq[len(seq)-num_aa:])
    
    def vectorize_sequences(self, sequence_array, extra_features=None):

        # Vectorize
        #vectorize_on_ss = np.vectorize(self.ss_features, signature='()->(k)')# This was trial
        vectorize_on_length = np.vectorize(len)
        vectorize_on_seq = np.vectorize(self.seq_features, signature='()->(k)')
        vectorize_on_seq_cat6 = np.vectorize(self.seq_cat6_features, signature='()->(k)')
        vectorize_on_get_nterm_seq = np.vectorize(self.get_nterm_seq)
        vectorize_on_get_cterm_seq = np.vectorize(self.get_cterm_seq)

        # Get vertorized output
        #ss_vector = vectorize_on_ss(sequence_array)#This was trial
        len_vector = np.reshape(vectorize_on_length(sequence_array), (-1, 1))
        seq_vector = vectorize_on_seq(sequence_array)
        seq_cat6_vector = vectorize_on_seq_cat6(sequence_array)
        
        sequence_array_nterm = vectorize_on_get_nterm_seq(sequence_array)
        nterm_seq_vector = vectorize_on_seq(sequence_array_nterm)

        sequence_array_cterm = vectorize_on_get_cterm_seq(sequence_array)
        cterm_seq_vector = vectorize_on_seq(sequence_array_cterm)

        
        #print(len_vector[:5])
        #print(ss_vector[:5])
        #print(seq_vector[:5])
        
        #print("Shapes of individual vectors:",
        #      len_vector.shape, seq_vector.shape, seq_cat6_vector.shape,
        #      nterm_seq_vector.shape, cterm_seq_vector.shape)
        
        # Concatenate features into one array
        fv = np.concatenate((len_vector, seq_vector, seq_cat6_vector, nterm_seq_vector, cterm_seq_vector), axis=1)
        
        # Concatenate extra features to the feature vector
        if extra_features is not None:
            #print(extra_features.shape)
            fv = np.concatenate((fv, extra_features.to_numpy()), axis=1)

        #print("Shape of feature vector:", fv.shape)
        #print(fv[:5])
        
        return(fv)

    def parameter_tuning(self, model, X, y, distributions, mode='RANDOM_SEARCH', verbose=0):
        if mode == 'RANDOM_SEARCH':
            mdl = RandomizedSearchCV(estimator=model, param_distributions=distributions, verbose=verbose, n_jobs=4, random_state=62)
            search = mdl.fit(X, y)
        
        if mode == 'GRID_SEARCH':
            mdl = GridSearchCV(estimator=model, param_grid=distributions, verbose=verbose, n_jobs=4)
            search = mdl.fit(X, y)
            
        return(search)

    def train(self, df_train, extra_features=None, mode='TRAIN', search_mode='RANDOM_SEARCH', model_type='DecisionTreeRegressor', verbose=0):
        log = open('train_'+model_type+'.log', 'w')
        
        X = self.vectorize_sequences(df_train['sequence'].to_numpy(), extra_features=extra_features)
        y = df_train['mean_growth_PH'].to_numpy()

        log.write("X.shape =" + str(X.shape))
        log.write("\n\n")


        # TRY DIFFERENT MODELS
        if model_type == 'DecisionTreeRegressor':
            model = tree.DecisionTreeRegressor()
            distributions = dict(max_depth = [2,3,5,7],
                                 min_samples_split = [2,5],
                                 min_samples_leaf = [1,3,5],
                                 max_features = ['auto','sqrt']
                                )
				
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=62)
            distributions = dict(n_estimators = [50,100,200,300],
                                 max_depth = [2,5,7,13],
                                 min_samples_split = [2,5],
                                 min_samples_leaf = [1,3,5,9],
                                 max_features = ['auto','sqrt'],
                                 oob_score=[True],
                                 random_state=[62]
                                )

        if model_type == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(random_state=62)
            distributions = dict(n_estimators = [50,100,200,300],
                                 max_depth = [2,3,5,7],
                                 min_samples_split = [2,5],
                                 min_samples_leaf = [1,3,5,9],
                                 max_features = ['auto','sqrt'],
                                 random_state=[62]
                                )

                
        if mode == 'TUNE' or mode == 'TUNE_AND_TRAIN':
            search = self.parameter_tuning(model, X, y, distributions, mode=search_mode, verbose=verbose)
            model = search.best_estimator_

            log.write("best_estimator_:\n")
            log.write(str(search.best_estimator_))
            log.write("\n\n")
            log.write("cv_results_:\n")
            log.write(str(search.cv_results_))
            log.write("\n\n")
            log.write("best_params_:\n")
            log.write(str(search.best_params_))
            log.write("\n\n")
            log.write("best_score_:\n")
            log.write(str(search.best_score_))
            log.write("\n\n")
        
        
        if mode == 'TRAIN' or mode == 'TUNE_AND_TRAIN':
            model.fit(X, y)
            score = model.score(X, y)

            log.write("Trained model score =" + str(score))
            log.write("\n\n")

            with open(self.model_file_path, 'wb') as model_file:
                pickle.dump(model, model_file)

        log.close()

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        return model.predict(X_vectorized)
