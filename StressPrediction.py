

import pandas as pd
import pandas
import scipy
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import collections
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import itertools
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

class StressPrediction:

    def __init__(self):
        self.dataset = pd.read_csv('/home/eltonss/PycharmProjects/flowPredicaoEstresse/Dataset/out.csv', index_col=0)
        self.dataset.drop(columns=['begin', 'end',], inplace=True)
        #print(self.dataset.shape)
        values_drop =['ignored','transient','meditation','amusement']
        self.dataset = self.dataset[~self.dataset['label'].isin(values_drop)]


    def preprocessing(self, column_name_target,split_only_training=False):
        # I'm codifying categorical value to numeric value
        le_sex = preprocessing.LabelEncoder()
        new = self.dataset.copy()
        le_sex.fit(list(set(self.dataset[column_name_target])))
        #print(Counter(new[column_name_target]))
        new[column_name_target] = le_sex.transform(new[column_name_target])
        #print(Counter(new[column_name_target]))
        # To convert for array
        y = np.asarray(new[column_name_target])
        df = self.dataset.copy().drop(column_name_target, axis=1)  # Remove the predict variable
        X = np.asarray(df)
        # Data Standardization give data zero mean and unit variance,
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        if(split_only_training):
            print('Train set:', X.shape, y.shape)
            return X,y;
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4,stratify=y)
        print('Train set:', X_train.shape, y_train.shape)
        print('Test set:', X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test;

    def plot_confusion_matrix(self, X_test,y_test,model, normalize=True, title='Confusion matrix', cmap='Blues'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = confusion_matrix(y_test, model.predict(X_test), labels=range(0, len(set(self.dataset['label']))))
        np.set_printoptions(precision=2)
        classes = sorted(Counter(self.dataset['label']).keys());

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm * 100
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        # Plot non-normalized confusion matrix
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=9)
        plt.yticks(tick_marks, classes, fontsize=9)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt) + "%", fontsize=10,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.show()

    def svc_param_selection(self,X, y, nfolds):
        Cs =     [0.01,0.1,1]
        gammas = [ 1,2,3,4,]
        decision_function_shape = ['ovo', 'ovr']
        param_grid = {'C': Cs,
                      'gamma' : gammas,
                      'degree':[1,2,3,4],
                      'kernel' : ['poly','sigmoid','rbf'],
                      'decision_function_shape': decision_function_shape
                      }
        grid_search = GridSearchCV(svm.SVC(probability=True), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        return grid_search,grid_search.best_params_

    # def featureSelection(self,column_name_target,n_best_feature=None):
    #     print('<feature Selection>')
    #     y = np.asarray(self.dataset[column_name_target])
    #     df = self.dataset.copy().drop(column_name_target, axis=1)  # Remove the predict variable
    #     number_feature_max = df.shape[1]
    #     X = np.asarray(df)
    #
    #     # Calculating scores #most recommended
    #     rank = fisher_score.fisher_score(X, y)
    #     feat_importances = pd.Series(rank, df.columns)
    #     feat_importances.name = 'rank'
    #     if (n_best_feature != None) and (n_best_feature<=number_feature_max):
    #
    #         result= feat_importances.sort_values(ascending=False).to_frame().reset_index()
    #         best_feature = list(result['index'].iloc[:n_best_feature])
    #         best_feature.append(column_name_target)
    #         print("The best features selected: ",best_feature)
    #         self.dataset=self.dataset[best_feature]
    #     feat_importances.plot(kind='barh', color='teal')
    #     plt.show()

    def featureSelectionRFECV(self,column_name_target,n_best_feature=None):
        print('<feature Selection>')
        dataframe = self.dataset.copy()
        dataframe[column_name_target] = dataframe[column_name_target].astype('category').cat.codes

        y = np.asarray(dataframe[column_name_target])
        df = dataframe.copy().drop(column_name_target, axis=1)  # Remove the predict variable
        X = np.asarray(df)

        # Instantiate estimator and feature selector
        svr_mod = SVR(kernel="linear")
        feat_selector = RFECV(svr_mod, cv=5)
        # Fit
        feat_selector = feat_selector.fit(X, y)
        # Print support and ranking
        print(feat_selector.support_)
        print(feat_selector.ranking_)
        print(df.columns)
        dataframe = pd.DataFrame({'features': df.columns, 'Values': feat_selector.ranking_})
        dataframe = dataframe.sort_values(by=['Values'], ascending=False)
        print(dataframe)
        best_feature = list(dataframe['features'].iloc[:n_best_feature])
        best_feature.append(column_name_target)
        print("The best features selected: ", best_feature)
        self.dataset = self.dataset[best_feature]

    def featureSelectionTree(self,column_name_target,n_best_feature=None):
        print('<feature Selection>')
        dataframe = self.dataset.copy()
        dataframe[column_name_target] = dataframe[column_name_target].astype('category').cat.codes

        y = np.asarray(dataframe[column_name_target])
        df = dataframe.drop(column_name_target, axis=1)  # Remove the predict variable
        X = np.asarray(df)

        # Instantiate
        rf_mod = ExtraTreesRegressor()
        # Fit
        rf_mod.fit(X, y)
        # Print
        #print(df.columns)
        #print(rf_mod.feature_importances_)

        #Selecting Features
        dataframe = pd.DataFrame({'Features':df.columns,'Values': rf_mod.feature_importances_})
        dataframe = dataframe.sort_values(by=['Values'],ascending=False)
        #print(dataframe)
        self.best_feature = list(dataframe['Features'].iloc[:n_best_feature])

        print("The best features selected: ", self.best_feature+[column_name_target])
        self.dataset = self.dataset[self.best_feature+[column_name_target]]


    def correlation(self,column_name_target_discrete=None):
        print("<Correlation>")


        dataframe = self.dataset.copy()
        dataframe[column_name_target_discrete] = dataframe[column_name_target_discrete].astype('category').cat.codes

        top = cm.get_cmap('Oranges_r', 128)
        bottom = cm.get_cmap('Blues', 128)

        newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                               bottom(np.linspace(0, 1, 128))))
        newcmp = ListedColormap(newcolors, name='OrangeBlue')
        corr = dataframe.corr()
        # Create positive correlation matrix
        corr = dataframe.corr().abs()

        # Correlation with output variable
        cor_target = corr[column_name_target_discrete]
        # Selecting highly correlated features
        best_features = cor_target[cor_target > 0.5]
        print(best_features)
        # Create and apply mask
        mask = np.triu(np.ones_like(corr, dtype=bool))
        tri_df = corr.mask(mask)
        # Find columns that meet treshold
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.9)]
        print(to_drop)



        reduced_df = dataframe.drop(to_drop, axis=1)
        print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1],reduced_df.shape[1]))  # Insert Column without erro

        # Create and apply mask
        mask = np.triu(np.ones_like(dataframe.corr(), dtype=bool))
        sns.heatmap(dataframe.corr(), mask=mask,center=0, cmap=newcmp, linewidths=1,annot=True, fmt=".2f")

        #Atualiza dataset
        self.dataset.drop(columns=to_drop,inplace=True)
        plt.show()

    def calculateMissingValues(self):
        print("<Detect missing values>")

        missing_values = self.dataset.isnull().sum()
        percent_missing = ((missing_values / self.dataset.index.size) * 100)
        #print(percent_missing)

    def removeOutliers(self):
        print('<Remove Outliers>')
        numeric_cols = self.dataset.select_dtypes(include=[np.number])
        categoric_cols = self.dataset.select_dtypes(include=['object'])
        # Criar index para manter as linhas abaixo do threshold
        # threshold de +/- 3 of SD
        idx = (np.abs(stats.zscore(numeric_cols)) < 3).all(axis=1)
        # Concatenate numeric and categoric subsets
        ld_out_drop = pd.concat([categoric_cols.loc[idx],numeric_cols.loc[idx]], axis=1)
        # atualiza o dataset
        print("Dimensionality reduced from {} to {}.".format(self.dataset.shape[0],
                                                             ld_out_drop.shape[0]))  # Insert Column without erro
        self.dataset = self.dataset.loc[idx]

if __name__ == '__main__':
    obj = StressPrediction()
    #print(obj.dataset.info())


    # Passo 1: Missiing Values
    obj.calculateMissingValues()

    #Passo 2: Detectar outliers
    obj.removeOutliers()

    #fig, ax = plt.subplots(1, 2)
    #sns.boxplot(x='label', y=ld_out_drop['Mean'],data=ld_out_drop,ax=ax[0])
    #sns.boxplot(x='label', y=obj.dataset['Mean'], data=obj.dataset, ax=ax[1])
    #plt.show()

    #Passo 3: Feature Selection
    #obj.correlation(column_name_target_discrete='label')
    obj.featureSelectionTree('label',7)

    #Passo 4: Dividi o conjunto de dados
    X_train, X_test, y_train, y_test = obj.preprocessing('label')

    #Passo 5: Executando Modelo
    model = svm.SVC(probability=True)
    best_params_ ={'C': 1, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 2, 'kernel': 'rbf'}
    model.set_params(**best_params_)
    model.fit(X_train, y_train)
    print('Accuracy: %.2f' % (model.score(X_test, y_test)))
    obj.plot_confusion_matrix(X_test, y_test, model)

    # Passo Especial: Obtendo melhor configuração
    #model,best_params_ = obj.svc_param_selection(X_train,y_train,3)
    #obj.plot_confusion_matrix(X_test, y_test, model)

