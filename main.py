# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import collections
import numpy as np
import matplotlib.pyplot as plt
import datetime

# import data from included examples
from pyphysio import EvenlySignal
# import all pyphysio classes and methods
import pyphysio as ph
import pyphysio.filters.Filters as flt

import os
import glob

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass;
    # obj = StressPrediction()
    # print(obj.dataset.info())
    # # separate array into input and output components
    # X_train, X_test, y_train, y_test = obj.preprocessing('label')
    #
    # train_accuracies = {}
    # test_accuracies = {}
    # neighbors = np.arange(1, 40)
    # for neighbor in neighbors:
    #     model = KNeighborsClassifier(n_neighbors=neighbor)
    #     model.fit(X_train, y_train)
    #     train_accuracies[neighbor] = model.score(X_train, y_train)
    #     test_accuracies[neighbor] = model.score(X_test, y_test)
    #
    # plt.figure(figsize=(8, 6))
    # plt.title("KNN: Variando Numero de vizinhos")
    # plt.plot(neighbors, train_accuracies.values(), label="Acuracia de Treinamento")
    # plt.plot(neighbors, test_accuracies.values(), label="Acuracia de teste")
    # plt.legend()
    # plt.xlabel("Vizinhos")
    # plt.ylabel("Accuracy")
    # plt.show()
    #
    # obj.plot_confusion_matrix(X_test, y_test, model)
    # print('Accuracy: %.2f' % (model.score(X_test, y_test)))
    ##################################################################
    # dataset = pd.read_csv('./Dataset/out.csv',index_col=0)
    #
    # dataset.drop(columns=['begin','end'],inplace=True)
    # print(dataset.info())
    ##############################################################
    # data = pd.read_pickle("./Dataset/S14.pkl")
    #
    # #print(data['label'])
    # # using Counter to find frequency of elements
    # frequency = collections.Counter(data['label'])
    # # create label
    # label = ph.EvenlySignal(data['label'], sampling_freq=700, signal_type='label')
    # # printing the frequency
    # print("Frequencia das classes")
    # print(dict(frequency))
    #
    # #print(len(data['label'])/700)
    # #print(data['signal']['wrist'])
    # #print(data['signal']['wrist']['EDA'])
    # #print(len(data['signal']['wrist']['EDA'])/4)
    #
    # # create signal
    # fsamp = 4 #4hz for Empatica
    # # assigned unix time
    # unix_time = 1495437325
    #
    # date_time = datetime.datetime.fromtimestamp(unix_time)
    #
    #
    # print("Seconds:=>",date_time.second)
    # tstart_eda = date_time.second
    # # displaying date and time in a regular
    # # string format
    # print("Date & Time =>",date_time.strftime('%Y-%m-%d %H:%M:%S'))
    #
    # eda_data = data['signal']['wrist']['EDA']
    #
    # eda = EvenlySignal(values=eda_data,
    #                    sampling_freq=fsamp,
    #                    signal_type='eda',
    #                    start_time=tstart_eda)
    #
    # print('EDA')
    # print('Sampling frequency: {}'.format(eda.get_sampling_freq()))
    # print('Start time:         {}'.format(eda.get_start_time()))
    # print('End time:           {}'.format(eda.get_end_time()))
    # print('Duration:           {}'.format(eda.get_duration()))
    # print('Signal type  :      {}'.format(eda.get_signal_type()))
    # print('First ten instants: {}'.format(eda.get_times()[0:10]))
    #
    # # resampling : decrease the sampling frequency by cubic interpolation
    # eda = eda.resample(fout=8, kind='cubic')
    # # IIR filtering : remove high frequency noise
    # eda_filt = ph.IIRFilter(fp=0.8, fs=1.1, ftype='ellip')(eda)
    # #eda.plot()
    # #eda_filt.plot()
    #
    # eda = eda_filt
    # # estimate the driver function
    # driver = ph.DriverEstim()(eda)
    #
    # # compute the phasic component
    # phasic, tonic, _ = ph.PhasicEstim(delta=0.01)(driver)
    # #eda.plot()
    # #phasic.plot()
    # #plt.show()
    #
    # # define a list of indicators we want to compute
    # indicators = [ph.Mean(),  ph.AUC(),
    #               ph.PeaksMean(delta=0.01,name='PeaksMean'),
    #               ph.DurationMean(delta=0.02,name='DurationMean')
    #               ]
    #
    # # define the windowing method
    # fixed_length = ph.FixedSegments(step=10, width=30,labels = label)
    #
    # # compute
    # PHA_ind, col_names = ph.fmap(fixed_length, indicators, phasic)
    # print(col_names)
    # PHA_ind_df = pd.DataFrame(PHA_ind, columns=col_names)
    #
    # print(PHA_ind_df)
    #
    # PHA_ind_df['label'] = PHA_ind_df['label'].replace([0,1,2,3,4,5,6,7],['transient','baseline','stress','amusement',
    #                                                                      'meditation','ignored','ignored','ignored'])
    # #PHA_ind_df['label'] = PHA_ind_df['label'].replace([2],['Estressada'])
    # boxplot = PHA_ind_df.boxplot(column=['PeaksMean', 'DurationMean' ],by='label')
    # plt.show()
    # # create a pandas dataframe

