from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split


import catboost
print('catboost version:', catboost.__version__)

df_train = pd.read_csv('/content/drive/My Drive/Janatahack_Cross_Sell_Prediction/train.csv')
df_test = pd.read_csv('/content/drive/My Drive/Janatahack_Cross_Sell_Prediction/test.csv')
df_sub = pd.read_csv('/content/drive/My Drive/Janatahack_Cross_Sell_Prediction/sample_submission.csv')

df_train.drop(['id'], axis = 1, inplace= True)
df_test.drop(['id'], axis = 1, inplace= True)

df_train.isnull().sum()
target = 'Response'

# Feature Engineering
df_train = df_train.replace({
    'Vehicle_Age':{'< 1 Year': 0,'1-2 Year' : 2, '> 2 Years': 3},
    'Vehicle_Damage':{'Yes': 0,'No' : 1},
    'Gender':{'Male': 0,'Female' : 1} 
    })

df_test = df_test.replace({
    'Vehicle_Age':{'< 1 Year': 0,'1-2 Year' : 2, '> 2 Years': 3},
    'Vehicle_Damage':{'Yes': 0,'No' : 1},
    'Gender':{'Male': 0,'Female' : 1}
    })

df_train['channel_premium_sum'] = df_train.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('sum')
df_train['channel_premium_max'] = df_train.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('max')
df_train['channel_premium_min'] = df_train.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('min')
df_train['channel_premium_mean'] = df_train.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('mean')
df_train['License_Age_Premium'] = df_train.groupby(['Driving_License','Age'])['Annual_Premium'].transform('sum')
df_train['VehicleAge_Premium'] = df_train.groupby(['Vehicle_Age'])['Annual_Premium'].transform('sum')
df_train['Age_channel'] = df_train.groupby(['Age','Policy_Sales_Channel'])['Annual_Premium'].transform('sum')

df_test['channel_premium_sum'] = df_test.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('sum')
df_test['channel_premium_max'] = df_test.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('max')
df_test['channel_premium_min'] = df_test.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('min')
df_test['channel_premium_mean'] = df_test.groupby(['Policy_Sales_Channel'])['Annual_Premium'].transform('mean')
df_test['License_Age_Premium'] = df_test.groupby(['Driving_License','Age'])['Annual_Premium'].transform('sum')
df_test['VehicleAge_Premium'] = df_test.groupby(['Vehicle_Age'])['Annual_Premium'].transform('sum')
df_test['Age_channel'] = df_test.groupby(['Age','Policy_Sales_Channel'])['Annual_Premium'].transform('sum')

df_train['Gender'] = df_train['Gender'].astype('category')
df_train['Age'] = df_train['Age'].astype('category')
df_train['Driving_License'] = df_train['Driving_License'].astype('category')
df_train['Region_Code'] = df_train['Region_Code'].astype('category')
df_train['Previously_Insured'] = df_train['Previously_Insured'].astype('category')
df_train['Vehicle_Age'] = df_train['Vehicle_Age'].astype('category')
df_train['Vehicle_Damage'] = df_train['Vehicle_Damage'].astype('category')
df_train['Policy_Sales_Channel'] = df_train['Policy_Sales_Channel'].astype('category')
df_train['Response'] = df_train['Response'].astype('category')



df_test['Gender'] = df_test['Gender'].astype('category')
df_test['Age'] = df_test['Age'].astype('category')
df_test['Driving_License'] = df_test['Driving_License'].astype('category')
df_test['Region_Code'] = df_test['Region_Code'].astype('category')
df_test['Previously_Insured'] = df_test['Previously_Insured'].astype('category')
df_test['Vehicle_Age'] = df_test['Vehicle_Age'].astype('category')
df_test['Vehicle_Damage'] = df_test['Vehicle_Damage'].astype('category')
df_test['Policy_Sales_Channel'] = df_test['Policy_Sales_Channel'].astype('category')

df_train.info()

lstColScale = [numcol for numcol in df_train.columns if df_train.dtypes[numcol] == 'int64' or df_train.dtypes[numcol] == 'float64']
lstColScale
df_train = fun_standardScaler(df_train, lstColScale)
df_test = fun_standardScaler(df_test, lstColScale)

df_train.info()

df_train['Response'] = df_train['Response'].astype('category')
x_train = df_train.drop([target], axis=1)
y_train = df_train[target]

import imblearn
from imblearn.over_sampling import SMOTE 

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
  
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel()) 
  
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

X_train_res_df = pd.DataFrame(X_train_res)
y_train_res_df = pd.DataFrame(y_train_res)

X_train_res_df.columns = ['Gender',
 'Age',
 'Driving_License',
 'Region_Code',
 'Previously_Insured',
 'Vehicle_Age',
 'Vehicle_Damage',
 'Policy_Sales_Channel',
 'Annual_Premium',
 'Vintage',
 'channel_premium_sum',
 'channel_premium_max',
 'channel_premium_min',
 'channel_premium_mean',
 'License_Age_Premium',
 'VehicleAge_Premium',
 'Age_channel']

y_train_res_df.columns = ['Response']


df_train_smote = pd.concat([X_train_res_df, y_train_res_df], axis=1)
df_train = df_train_smote.copy()

df_train.info()

df_train['Response'] = df_train['Response'].astype('category')
X = df_train.drop([target], axis=1)
y = df_train[target]

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

feats = list(df_train)
feats.remove('Response')
print(feats)

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

pred = fun_SKFold_Binary_ClassificationAll(df_train, target)

df_pred = pd.DataFrame(pred)
df_pred.columns = ['Response']
df_sub['Response'] = df_pred['Response']
df_sub

df_sub.groupby(['Response'])['Response'].count()

df_sub.to_csv("/content/drive/My Drive/Janatahack_Cross_Sell_Prediction/GNB_Balanced2.csv", header=True, index = False)

# StratifiedKFold - based on the defined splits all  model will be executed and the probability predictions for all levels of target are stored in oof_preds.
# Accuracy is calculated for each of the folds and is saved in final_preds which is averaged out in the end for final accuracy score

def fun_SKFold_Binary_ClassificationAll(df, target):
  import sklearn.metrics as metrics
  from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split,StratifiedShuffleSplit
  from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.naive_bayes import GaussianNB
  from sklearn.metrics import roc_auc_score,accuracy_score ,confusion_matrix, f1_score, precision_score, recall_score
  from catboost import CatBoostClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import roc_curve
  from sklearn import tree
  import lightgbm as lgb
  from xgboost import XGBClassifier
  splits = 5
  levels = df[target].nunique()
  folds =StratifiedKFold(n_splits=splits, random_state=22,shuffle=True)
  oof_preds = np.zeros((len(df_test), levels))
  #feature_importance_df = pd.DataFrame()
  #feature_importance_df['Feature'] = X_train.columns
  final_preds = []
  random_state = [22,44,66,77,88,99,101]
  counter = 0
  num_model = 1

  for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train)):
          #print("Fold {}".format(fold_))
          X_trn,y_trn = X_train[feats].iloc[trn_idx],y_train.iloc[trn_idx]
          X_val,y_val = X_train[feats].iloc[val_idx],y_train.iloc[val_idx]

          def fun_metrics(predictions, y_val):
              print("Confusion Matrix----------------------")
              cm = confusion_matrix(predictions,y_val)
              print(cm)
              f1Score = f1_score(y_val, predictions, average='weighted')
              print("F1 Score :",f1Score)
              f1Score = None
              precision = precision_score(y_val, predictions)
              print("Precision :", precision)
              recall = recall_score(y_val, predictions)
              print("Recall :", recall)
              

          def fun_metricsPlots(fpr, tpr,model):
              fpr, tpr, _ = roc_curve(y_val, predictions)
              plt.plot(fpr, tpr, linestyle='--', label= model)
              print("AUC :", metrics.auc(fpr, tpr))

          def fun_updateAccuracy(model, predictions):
              global oof_preds
              final_preds.append(accuracy_score(y_pred=predictions,y_true=y_val))
              modelAccuracy = accuracy_score(predictions,y_val)
              print("Model Accuracy: ", modelAccuracy)
                            
    
          print("Executing Gaussian Naive Bayes  for fold#:", fold_)
          clf = GaussianNB()
          clf.fit(X_trn,y_trn)
          predictions = clf.predict(X_val)
          fun_metrics(predictions, y_val)
          fpr, tpr, _ = roc_curve(y_val, predictions)
          fun_metricsPlots(fpr, tpr, "GNB")
          fun_updateAccuracy(clf, predictions)
          print("==========================================")
          oof_preds += clf.predict_proba(df_test[feats])


          final_preds.append(accuracy_score(y_pred=clf.predict(X_val),y_true=y_val))
          #final_preds.append(metrics.auc(fpr, tpr))
          oof_preds += clf.predict_proba(df_test[feats])
                  

          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.legend()
          plt.show()
          
  oof_preds  = oof_preds/splits/num_model
  predictions_sub = [np.argmax(x) for x in oof_preds]
  ############print("Predictions for submission: ", predictions_sub )
  print("##################################")
  print("Average Accuracy :",sum(final_preds)/len(final_preds))
  print("All accuracy: ", final_preds)
  print("##################################")

  c1 = np.transpose(final_preds)
  d1 = np.array_split(c1,len(c1)/splits)
  d2 = pd.DataFrame(d1)

  #modelNames = pd.DataFrame(['CATBOOST','XGBOOST','LGBM','RF','KNN','GNB','DT','ADA','GLM'])
  modelNames = pd.DataFrame(['XGBOOST'])
  modelName1 = pd.concat([modelNames, d2], axis = 1)
  modelName1.columns = ['Model Name','Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']
  modelName1['Avg Accuracy'] = modelName1.mean(axis=1)
  modelName1.sort_values(by = 'Avg Accuracy', ascending= False, inplace= True)
  print(modelName1)
  return predictions_sub

"""# **Functions**"""

####################################################################################################################################################################################
#catboost
####################################################################################################################################################################################

#Check for class imbalance
def fun_classImbalanceCheck(df,target):
    return df.groupby([target]).size()/len(df)*100, sns.countplot(target,data=df)

# For all categorical varibles in train and test - check for levels
def fun_levelCheck(df_train, df_test):
  cols =  list(df_test.select_dtypes(include= ['object']).columns)
  if cols != 'Segmentation':
    for col in cols:
      print('Total unique '+col  +' values in Train are {}'.format(df_train[col].nunique()))
      print('Total unique '+col  +' values in Test are {}'.format(df_test[col].nunique()))
      print('Common'+col +' values are {}'.format(len(list(set(df_train[col]) & set(df_test[col])))))
      print('**************************')

#Split the data in to train and test. Check for spit numbers
def fun_split(df):
  X = df.drop([target], axis=1)
  y = df[target]
  
  from sklearn.model_selection import train_test_split
  X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

  # describes info about train and test set 
  print("Number transactions X_train dataset: ", X_train.shape) 
  print("Number transactions y_train dataset: ", y_train.shape) 
  print("Number transactions X_validation dataset: ", X_validation.shape) 
  print("Number transactions y_validation dataset: ", y_validation.shape) 
  return X_train, X_validation, y_train, y_validation

#Categorical encoding which calculates the percentage contribution of each of the categorical variables
def frequency_encoding(df_Train):
  for col in list(df_train.select_dtypes(include=['float64']).columns):
      if col!='Segmentation':
          fe=df_train.groupby(col).size()/len(df_train)
          df_train[col]=df_train[col].apply(lambda x: fe[x])
  return df_train

# Data split in to train and validation. model building. predictio on test data and preparation of submission file
def fun_catboost(X,y, X_train,X_validation, y_train, y_validation,target):
  
  #Creating a training set for modeling and validation set to check model performance
  #X = df_train.drop(['Segmentation', 'Gender','Ever_Married', 'Work_Experience','Family_Size','Var_1'], axis=1)

  #categorical_features_indices = np.where(df_train.dtypes != np.float)[0]
  categorical_features_indices = list(range(len(X_train.columns)))
  categorical_features_indices

  #importing library and building model
  from catboost import CatBoostClassifier
  model=CatBoostClassifier(iterations=5, depth=3, learning_rate=0.1, loss_function='MultiClass', eval_metric='Accuracy')
  model.fit(X_train, y_train,eval_set=(X_validation, y_validation),plot=True)

  predictions = model.predict(df_test)

  model.get_feature_importance(type= "FeatureImportance")
  from catboost import Pool, CatBoostClassifier
  from catboost.utils import get_confusion_matrix

  train_label = ["A", "B", "C", "D"]
  cm = get_confusion_matrix(model, Pool(X_validation, y_validation))
  print(cm)
  print(model.get_best_score())

  submission = pd.DataFrame()
  submission['ID'] = df_test['ID']
  submission[target] = predictions
  return categorical_features_indices, model, submission, predictions

# StratifiedKFold - based on the defined splits the chosen model will be executed and the probability predictions for all levels of target are stored in oof_preds.
# Accuracy is calculated for each of the folds and is saved in final_preds which is averaged out in the end for final accuracy score



def fun_varImp(model, X_train):

    print(model.feature_importances_)
    names = X_train.columns.values
    ticks = [i for i in range(len(names))]
    plt.bar(ticks, model.feature_importances_)
    plt.xticks(ticks, names,rotation =90)
    plt.show()

#Variable importance extraction using permutation
def fun_PermutationImportance(X_validation,y_validation):
  perm = PermutationImportance(model,random_state=100).fit(X_validation, y_validation)
  return eli5.show_weights(perm,feature_names=X_validation.columns.tolist())

####################################################################################################################################################################################
#XGBosst
####################################################################################################################################################################################

#Dummify the categorical variables
def fun_dummify(df):
  for col in list(df.select_dtypes(include=['category']).columns):
    df = pd.get_dummies(df,columns=list(df.select_dtypes(include=['category']).columns),drop_first=True)
  return df

# Data split in to train and validation. model building. predictio on test data and preparation of submission file
def fun_xgboost(target):
  import xgboost as xgb
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import LabelEncoder 
  from xgboost import XGBClassifier
  from sklearn.metrics import accuracy_score


  X = df_train.drop([target], axis=1)
  y = df_train[target]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
  lc = LabelEncoder() 
  lc = lc.fit(y) 
  lc_y = lc.transform(y)

  model = XGBClassifier() 
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test) 
  predictions = [value for value in y_pred]
 
  accuracy = accuracy_score(y_test, predictions) 
  print("##################################")
  print("Accuracy: %.2f%%" % (accuracy * 100.0))
  print("##################################")




# Plot for all categorical variables from the dataframe
def fun_catPlots(df):
  cols =  list(df.select_dtypes(include= ['object']).columns)
  for col in cols:
    plt.figure(figsize=(10,5))
    ax = sns.countplot(x = col, data = df, order= df[col].value_counts().index) # hue = col - to get legends
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Create new polynomial features for selected columns. The function accepts the dataset and column list for processing, merges it with orignial dataset and returns the dataframe

def fun_polynomialFeature(df, lstColname):

  #Use Sklearn Polynomial Features
  from sklearn.preprocessing import PolynomialFeatures

  poly = PolynomialFeatures()
  #to_cross = ['age', 'length_of_service', 'avg_training_score']
  to_cross = lstColname
  crossed_feats = poly.fit_transform(df[to_cross].values)

  #Convert to Pandas DataFrame and merge to original dataset
  crossed_feats = pd.DataFrame(crossed_feats)
  crossed_feats = crossed_feats.rename(columns= lambda x: "NF"+str(x)) # prefix all the new variables with "NF"
  df = pd.concat([df, crossed_feats], axis=1)
  return df

# Apply Standard scaler on the selected column. The function accepts dataset and list of columns for processing. It deletes the columns in the main dataset and updates it with scaled columns with same name

def fun_standardScaler(df, lstColname):
    from sklearn.preprocessing import StandardScaler

    feats = lstColname
    sc = StandardScaler()
    sc_data = sc.fit_transform(df[feats])

    cnt = 0
    for i in feats:
      df.drop(feats[cnt], axis=1, inplace= True)
      cnt += 1
      
    sc_data = pd.DataFrame(sc_data)
    sc_data.columns = feats
    cnt = 0
    df = pd.concat([df, sc_data], axis=1)
    return df

def fun_corrPlot(df):

    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True)

def fun_dataExplore(df_train, df_test):
      
    #missing data
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)*100
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)
    print("====================================================")
    print(df_train.info())
    print("====================================================")
    print(fun_levelCheck(df_train, df_test))

    print("================= Test Data =====================")
    #missing data
    total = df_test.isnull().sum().sort_values(ascending=False)
    percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)*100
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)
    print("====================================================")
    print(df_test.info())
    print("====================================================")

def fun_numPlots(df, lstColname):
    
    # plots with raw data before transformation
    for i, col in enumerate(lstColname):
        sns.distplot(df[lstColname[i]])
        plt.title("Histogram of: "+col)
        plt.show()

        # log transformed plots
        sns.distplot(np.log1p(df[lstColname[i]]))
        plt.title("Histogram of log transformed: "+col)
        plt.show()
        print("========================================================================")
    df[lstColname].hist(bins=15, figsize=(15, 6), layout=(2, 4));
    np.log1p(df[lstColname]).hist(bins=15, figsize=(15, 6), layout=(2, 4));

def fun_featureImportance(X,y):
    from sklearn.ensemble import RandomForestClassifier
    reg = RandomForestClassifier(n_estimators=50)
    reg.fit(X, y)

    df_feature_importance = pd.DataFrame(reg.feature_importances_, index= X.columns, columns=['feature importance']).sort_values('feature importance', ascending=True)
    df_feature_importance.plot(kind='barh');
    plt.rcParams["figure.figsize"] = [8,20]

# function to get cross validation scores
def fun_get_cv_scores(model):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print("==========================================================================")
    print("Model: ", model)
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))

def fun_defineModels():
    global ridge,lasso, elastic,lasso_lars,bayesian_ridge,logistic,sgd
    # define models
    ridge = linear_model.Ridge()
    lasso = linear_model.Lasso()
    elastic = linear_model.ElasticNet()
    lasso_lars = linear_model.LassoLars()
    bayesian_ridge = linear_model.BayesianRidge()
    logistic = linear_model.LogisticRegression(solver='liblinear')
    sgd = linear_model.SGDClassifier()

    #models = [ridge, lasso, elastic, lasso_lars, bayesian_ridge, logistic, sgd]
    models = [ridge, lasso]

    # loop through list of models
    for model in models:
        fun_get_cv_scores(model)

def fun_GridSearchCV(model):
      penalty = ['l1', 'l2']
      C = [0.01, 0.1, 1, 10]
      class_weight = [{1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
      solver = ['liblinear']

      param_grid = dict(penalty=penalty,
                        C=C,
                        class_weight=class_weight,
                        solver=solver)

      grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)
      grid_result = grid.fit(X_train, y_train)

      print("==========================================================================")
      print(model,' Best Score: ', grid_result.best_score_)
      print(model,' Best Params: ', grid_result.best_params_)

def fun_RandomizedSearchCV(model):

      loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
      penalty = ['l1', 'l2', 'elasticnet']
      alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
      learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
      class_weight = [{1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
      eta0 = [1, 10, 100]

      param_distributions = dict(loss=loss,
                                penalty=penalty,
                                alpha=alpha,
                                learning_rate=learning_rate,
                                class_weight=class_weight,
                                eta0=eta0)

      random = RandomizedSearchCV(estimator=model,
                                  param_distributions=param_distributions,
                                  scoring='roc_auc',
                                  verbose=1, n_jobs=-1,
                                  n_iter=1000)
      random_result = random.fit(X_train, y_train)
      print("==========================================================================")
      print('Best Score: ', random_result.best_score_)
      print('Best Params: ', random_result.best_params_)

# check for relation of the missing values on the target.
def fun_missing_vs_target(df):
    features_with_na = [features for features in df.columns if df[features].isnull().sum()>1] 

    for feature in features_with_na:
                
        # let's make a variable that indicates 1 if the observation was missing or zero otherwise
        df[feature] = np.where(df[feature].isnull(), 1, 0)
        # let's calculate the mean SalePrice where the information is missing or present
        df.groupby(feature)[target].median().plot.bar()
        plt.title(feature)
        plt.show()

# Function to plot boxplot, pass the list of columns either continuus or discreet
def fun_outliers(df,lstcolumn):
    for feature in lstColname:
        if 0 in df[feature].unique():
            pass
        else:
            df[feature]=np.log(df[feature])
            df.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()







# def fun_combineDF(df_train, df_test):
#     df_train['split_Identifier'] = 'train'
#     df_test['split_Identifier'] = 'test'
#     df_combined = pd.concat([df_train, df_test], axis = 0)
#     return df_combined, df_train, df_test

# tempdf = fun_combineDF(df_train, df_test)

# df_combined = tempdf[0]
# df_train = tempdf[1]
# df_test = tempdf[2]

# Sequence of fucntion calls

# #---------EDA------------
# fun_dataExplore()
# fun_levelCheck()
# fun_classImbalanceCheck()
# fun_catPlots()

# #--------Data preprocessing-------------
# frequency_encoding()
# fun_dummify()
# fun_split()
# df = fun_polynomialFeature()
# df = fun_standardScaler()
# fun_featureImportance(X,y)

# #-------Model Building ----------------

# fun_SKFold_Binary_ClassificationAll()





