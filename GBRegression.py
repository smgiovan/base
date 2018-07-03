# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:23:01 2018

@author: sg867887
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, r2_score

    
def _getData():
        
    data = 'kaggle'
    
    if data == 'QorvoML':
        df = pd.read_csv('C:/Users/sg867887/Desktop/Qorvo_ML_Dataset.csv',sep='\t')
        df.dropna(inplace=True)
        features = df.iloc[:,5:]
        labels = df.iloc[:,1]
        Y = np.array(labels)
        X = np.array(features)
        Xnames = list(features.columns)

    elif data == 'diabetes':
        import sklearn.datasets as dat;
        dataset = dat.load_diabetes()
        X = dataset.data
        Y = dataset.target
        Xnames = dataset.feature_names
        Ynames = 'Diabetes Progression Index'

    elif data == 'kaggle':
        df = pd.read_csv('https://raw.githubusercontent.com/smgiovan/base/master/train.csv') 
        features = df.iloc[:,0:-1]
    
        features = features[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']]
    
        features = pd.get_dummies(features)
        
        labels = df.iloc[:,-1]
        Y = np.array(labels)
        X = np.array(features)
        Xnames = list(features.columns)
        Ynames = df.columns[-1]
    
    else:
        print('you need to define your dataset..')
    
    # Split the data into training and testing sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size = 0.25, random_state = 42)
    
    return Xtrain, Xtest, Ytrain, Ytest, Xnames, X, Y, Ynames

def rmsle(y0, y):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
 
def customScoreFunc(Yactual, Ypred):
    
    #return r2_score(Yactual,Ypred)
    return rmsle(Yactual, Ypred)

scorer = make_scorer(customScoreFunc,greater_is_better=False)




def getRegressor(params=None):
    # setup regressor with default parameters
    # for good explaination of parameters:
    #  see https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    
    rf = xgb.XGBRegressor()
    
    if params:
        rf.set_params(**params)
            
    return rf


def getRegressorParams():
    
    params = {
            'base_score': 0.5,
            'booster': 'gbtree',
            'colsample_bylevel': 1,
            'colsample_bytree': 1,
            'gamma': 0,
            'learning_rate': 0.1,
            'max_delta_step': 0,
            'max_depth': 3,
            'min_child_weight': 1,
            'missing': None,
            'n_estimators': 100,
            'n_jobs': 1,
            'nthread': None,
            'objective': 'reg:linear',
            'random_state': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'seed': None,
            'silent': True,
            'subsample': 1
            }
    
    return params

def getGridParams():
    
    param_grid = { 
                  'colsample_bytree' : [0.35,0.5,0.65],
                  'max_depth' : [2,3,4], 
                  'learning_rate': [1e-2,5e-2,1e-1],
                  'n_estimators': [50,100,150,300,500]
                  }
          
    #param_grid = { 
    #              'colsample_bytree' : [0.2,0.4,0.6,0.8,1],
    #              'max_depth' : [3,4,5,6,7], 
    #              'learning_rate': [5e-2,1e-1,2e-1,3.5e-1,5e-1],
    #              'n_estimators': [100,150,350,650,1000]
    #              }
    
    return param_grid

def GBgridSearch(Xtrain,Ytrain,regressor_params=None,param_grid={}):

    kfold = 3
    
    
    rf = getRegressor(regressor_params)

    grid_search = GridSearchCV(
            estimator = rf, param_grid = param_grid, 
            cv = kfold,
            n_jobs = 1, 
            verbose = 1,
            scoring=scorer
            )

    grid_search.fit(Xtrain, Ytrain)
    
    return grid_search
    

def plotFeatureImportance(feature_importances):
    fig_FeatImp, (ax_FeatImp) = plt.subplots()
    feature_importances[:10].plot(kind='bar', title='Feature Importances (top 10)', ax=ax_FeatImp)
    plt.ylabel('Feature Importance Score')
    plt.tight_layout()
    plt.show()
    
def plotTestResults(test_results,Ynames):
    fig_TestResults, (ax_TestResults) = plt.subplots()
    test_results.plot(
            x='ind',
            y='predicted',
            kind='scatter',c='red',ax=ax_TestResults)
    test_results.plot(
            x='ind',
            y='actual',
            kind='scatter', title='Actual vs Predicted', ax=ax_TestResults)
    ax_TestResults.get_xaxis().set_ticks([])
    ax_TestResults.set_xlabel("")
    ax_TestResults.set_ylabel(Ynames)
    ax_TestResults.legend(["Predicted", "Actual"])
    plt.tight_layout()
    plt.show()
    
 #%%   
if __name__ == '__main__':
        
    Xtrain, Xtest, Ytrain, Ytest, Xnames, X, Y, Ynames = _getData()
    
    grid_search = GBgridSearch(Xtrain,Ytrain,
                               regressor_params=getRegressorParams(),
                               param_grid=getGridParams()
                               )
    
    best_grid = grid_search.best_estimator_
    
    # summarize grid search
    print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print('')
    
    print("Score: {0:1.3f}".format(customScoreFunc(Ytest,best_grid.predict(Xtest))))
    
    print("Explained Variance(1-u/v): {0:1.3f}".format(best_grid.score(Xtest,Ytest)))
    
    print("Correlation Coeff.: %f" % np.corrcoef(Ytest,best_grid.predict(Xtest))[0,1])
    
    # get and plot feature importance
    feature_importances = pd.DataFrame({
                    'Feature' : Xnames,
                    'Importance' : best_grid.feature_importances_,
                    })
    feature_importances.set_index(["Feature"],inplace=True)
    feature_importances.sort_values(by='Importance',ascending=False,inplace=True)
    plotFeatureImportance(feature_importances)
    
    # get and plot test results
    test_results = pd.DataFrame({
                    'actual' : Ytest,
                    'predicted' : best_grid.predict(Xtest),
                    })
    test_results.sort_values(by='actual',ascending=False,inplace=True)
    test_results['ind'] = range(len(Ytest))
    plotTestResults(test_results,Ynames)

    # retrain model using all data
   # trained_model = GBgridSearch(X,Y,
   #                            regressor_params=best_grid.get_params(),
   #                            param_grid={}
   #                            )
    
    
    #xgb.plot_tree(best_grid,
    #              num_trees=best_grid.get_params(deep=True).get('n_estimators')-1,
    #              rankdir='LR')
    