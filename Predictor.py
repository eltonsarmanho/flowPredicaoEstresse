from StressPrediction import StressPrediction
from sklearn.svm import SVR
from sklearn import svm
# import data from included examples
from pyphysio import EvenlySignal
# import all pyphysio classes and methods
import pyphysio as ph
import pyphysio.filters.Filters as flt
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing

class Predictor:
    def __init__(self):
        pass;
    def load(self):
        obj = StressPrediction()
        # Passo 1: Missiing Values
        obj.calculateMissingValues()
        # Passo 2: Detectar outliers
        obj.removeOutliers()
        # Passo 3: Feature Selection
        obj.featureSelectionTree('label', 7)

        # Passo 4: Data Split
        X_train, y_train = obj.preprocessing('label',split_only_training=True)

        # Passo 5: training the model
        self.model = svm.SVC(probability=True)
        best_params_ ={'C': 1, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 2, 'kernel': 'rbf'}
        self.model.set_params(**best_params_)
        self.model.fit(X_train, y_train)
        self.model.best_feature = obj.best_feature

    def responsePrediction(self,dict_eda):

        eda_data = dict_eda['eda_data']
        fsamp  = dict_eda['fsamp']

        date_time = datetime.datetime.fromtimestamp(dict_eda['star_time'])
        star_time_eda = date_time.second
        #print(fsamp,star_time_eda)

        eda = EvenlySignal(values=eda_data,
                           sampling_freq=fsamp,
                           signal_type='eda',
                           start_time=star_time_eda)
        blocos = 4
        size_block_begin = 31
        duration  = size_block_begin + (blocos-1)*10
        eda = eda.segment_time(star_time_eda, star_time_eda+duration)
        print('EDA')
        print('Sampling frequency: {}'.format(eda.get_sampling_freq()))
        print('Start time:         {}'.format(eda.get_start_time()))
        print('End time:           {}'.format(eda.get_end_time()))
        print('Duration:           {}'.format(eda.get_duration()))
        print('Signal type  :      {}'.format(eda.get_signal_type()))
        print('First ten instants: {}'.format(eda.get_times()[0:10]))

        # resampling : decrease the sampling frequency by cubic interpolation
        eda = eda.resample(fout=8, kind='cubic')
        # IIR filtering : remove high frequency noise
        eda_filt = ph.IIRFilter(fp=0.8, fs=1.1, ftype='ellip')(eda)
        eda = eda_filt

        # estimate the driver function
        driver = ph.DriverEstim()(eda)

        # compute the phasic component
        phasic, tonic, _ = ph.PhasicEstim(delta=0.02)(driver)
        # define the windowing method
        fixed_length = ph.FixedSegments(step=10, width=30)

        # compute all features from EDA
        PHA_ind, col_names = ph.fmap(fixed_length, ph.preset_phasic(delta=0.02), phasic)
        col_names_update = list(map(lambda x: str(x).replace('pha_', ''), col_names))
        PHA_ind_df = pd.DataFrame(PHA_ind, columns=col_names_update)
        PHA_ind_df = PHA_ind_df[self.model.best_feature]
        X = np.asarray(PHA_ind_df)
        # Data Standardization give data zero mean and unit variance,
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

        return self.model.predict(X)



if __name__ == '__main__':
    model = Predictor()
    model.load()

    #Simulação
    data_eda = pd.read_csv('/home/eltonss/Documents/Flow/WESAD/S14/S14_E4_Data/EDA.csv', header=None)
    # Start time
    unix_time = data_eda.iloc[0, 0]
    # Frequency
    fsamp = data_eda.iloc[1, 0]
    # raw data
    eda_data = data_eda.iloc[2:, 0:].values
    #Dados em Dict
    dict_eda={'star_time':unix_time,'fsamp':fsamp,'eda_data':eda_data}

    answer = model.responsePrediction(dict_eda)
    print(answer)

