
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

class CreateDatasetEmotional:
    def __init__(self,rootdir):
        self.rootdir = rootdir
        pass;

    def getFilesPkl(self,rootdir,extensio):
        participants = []
        files_pkl = []
        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d):
                participants.append(os.path.basename(os.path.normpath(d)))
                files = os.path.join(d, extensio)
                files = glob.glob(files)
                files_pkl.append(files)
        return participants,files_pkl

    def getFilesEDA(self,rootdir,participants,extensio):
        files_eda = []
        for p in participants:
            dir = rootdir+p
            for file in os.listdir(dir):
                d = os.path.join(dir, file)
                if os.path.isdir(d):
                    files = os.path.join(d, extensio)
                    files = glob.glob(files)
                    files_eda.append(files)
        return files_eda

    def runConstructor(self,files_eda,files_pkl):
        #Ler dados EDA e PKL
        list_of_dataframes = []

        for file_eda,file_pkl in zip (files_eda,files_pkl):
            print(file_eda[0],file_pkl[0])
            data_pkl = pd.read_pickle(file_pkl[0])
            data_eda = pd.read_csv(file_eda[0],header=None)

            #Start time
            unix_time = data_eda.iloc[0,0]
            date_time = datetime.datetime.fromtimestamp(unix_time)
            start_eda = date_time.second

            #Frequency
            fsamp = data_eda.iloc[1,0]

            # using Counter to find frequency of elements
            frequency = collections.Counter(data_pkl['label'])
            # create label
            label = ph.EvenlySignal(data_pkl['label'], sampling_freq=700, signal_type='label')

            eda_data = data_eda.iloc[2:,:].values
            print(type(eda_data))
            eda = EvenlySignal(values=eda_data,
                               sampling_freq=fsamp,
                               signal_type='eda',
                               start_time=start_eda)

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

            # define a list of indicators we want to compute
            indicators = [ph.Mean(name='Mean'), ph.AUC(name='AUC'),
                          ph.PeaksMean(delta=0.02, name='PeaksMean'),
                          ph.DurationMean(delta=0.02, name='DurationMean')
                          ]

            # define the windowing method
            fixed_length = ph.FixedSegments(step=10, width=30, labels=label)

            # compute all features from EDA
            PHA_ind, col_names = ph.fmap(fixed_length,  ph.preset_phasic(delta=0.02), phasic)
            col_names_update = list(map(lambda x: str(x).replace('pha_', ''), col_names))
            PHA_ind_df = pd.DataFrame(PHA_ind, columns=col_names_update)

            PHA_ind_df['label'] = PHA_ind_df['label'].replace([0, 1, 2, 3, 4, 5, 6, 7],
                                                              ['transient', 'baseline', 'stress', 'amusement',
                                                               'meditation', 'ignored', 'ignored', 'ignored'])
            print(PHA_ind_df)
            list_of_dataframes.append(PHA_ind_df)
        #Save all dataframe in CSV
        df = pd.concat(list_of_dataframes)
        #df.to_csv(self.rootdir+'out.csv')

if __name__ == '__main__':
    rootdir = '/home/eltonss/Documents/Flow/WESAD/'
    obj = CreateDatasetEmotional(rootdir)
    participants,files_pkl = obj.getFilesPkl(rootdir,'*.pkl')
    print(participants,files_pkl)
    files_eda = obj.getFilesEDA(rootdir,participants, 'EDA*.csv')

    obj.runConstructor(files_eda,files_pkl)
