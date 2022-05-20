# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoDP: Automated Data Preparation for Machine Learning Models")
print("Creator: Mohammad Reza Saraei")
print("Contact: mrsaraei@yahoo.com")
print("Supervisor: Dr. Saman Rajebi")
print("Created Date: May 20, 2022")
print("") 

print("------------------------------------------------------")
print("------------------ Import Libraries ------------------")
print("------------------------------------------------------")
print("")

# Import Libraries for Python
import pandas as pd
import numpy as np
from pandas import set_option
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import Counter
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("------------------------------------------------------")
print("------------------ Data Ingestion --------------------")
print("------------------------------------------------------")
print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('RawData.csv')

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

print("------------------------------------------------------")
print("------------ Initial Data Understanding --------------")
print("------------------------------------------------------")
print("")

print("Initial General Information:")
print("****************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("---------------- Data Label Encoding -----------------")
print("------------------------------------------------------")
print("")

# Encoding Coulmns Having Objects by LabelEncoder
obj = df.select_dtypes(include = ['object'])
LE = preprocessing.LabelEncoder()
col = obj.apply(LE.fit_transform)

print("Columns Having Objects:")
print("***********************")
print(obj.head(10))
print("")

print("Encoding Columns Having Object:")
print("*******************************")
print(col.head(10))
print("")
print('Shape of Encoded Columns:', col.shape)
print("")

print("------------------------------------------------------")
print("------------- Save Encoded Objects Data --------------")
print("------------------------------------------------------")
print("")

# Save DataFrame After Encoding
pd.DataFrame(col).to_csv('EncodedData.csv', index = False)

print("------------------------------------------------------")
print("------- Creating Main DataFrame by Combination -------")
print("------------------------------------------------------")
print("")

# Import Encoded Objects DataFrame (.csv) by Pandas Library
df_col = pd.read_csv('EncodedData.csv')

# Combinating Encoded Data with Main DataFrame
df_obj = df.drop(df.select_dtypes(include = ['object']), axis = 1)

print("Columns' Name that needs to encoding:", obj.columns)
print("")

print("The Target Column Name:", df.columns[-1])
print("")

if df.columns[-1] in obj.columns:
    df = pd.concat([df_obj, df_col], axis = 1)
else:
    df = pd.concat([df_col, df_obj], axis = 1)

print("An overview of Encoded Data:")
print("****************************")
print("")
print(df.head(1))
print("")

print("------------------------------------------------------")
print("--------- Data Understanding After Encoding ----------")
print("------------------------------------------------------")
print("")

print("General Information After Encoding:")
print("***********************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("------------------ Data Spiliting --------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("---------------- Data Normalization ------------------")
print("------------------------------------------------------")
print("")

# Normalization [0, 1] of Data
scaler = MinMaxScaler(feature_range = (0, 1))
f = scaler.fit_transform(f)
print(f)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv('FeaturesData.csv', index = False)
pd.DataFrame(t).to_csv('TargetData.csv', index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv('FeaturesData.csv')
df_t = pd.read_csv('TargetData.csv')

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv('MainDataFrame.csv', index = False)

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('MainDataFrame.csv')

print("------------------------------------------------------")
print("---------------- Data Preprocessing ------------------")
print("------------------------------------------------------")
print("")

# Replace Question Mark to NaN:
df.replace("?", np.nan, inplace = True)

# Remove Duplicate Samples
df = df.drop_duplicates()
print("Duplicate Records After Removal:", df.duplicated().sum())
print("")

# Replace Mean instead of Missing Values
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df)
df = imp.transform(df)
print("Mean Value For NaN Value:", "{:.3f}".format(df.mean()))
print("")

# Reordering Records / Samples / Rows
print("Reordering Records:")
print("*******************")
df = pd.DataFrame(df).reset_index(drop = True)
print(df)
print("")

print("------------------------------------------------------")
print("------------------ Data Respiliting ------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("----------------- Outliers Detection -----------------")
print("------------------------------------------------------")
print("")

# Identify Outliers in the Training Data
ISF = IsolationForest(n_estimators = 100, contamination = 0.1, bootstrap = True, n_jobs = -1)

# Fitting Outliers Algorithms on the Training Data
ISF = ISF.fit_predict(f, t)

# Select All Samples that are not Outliers
Mask = ISF != -1
f, t = f[Mask, :], t[Mask]

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("------------- Data Balancing By SMOTE ----------------")
print("------------------------------------------------------")
print("")

# Summarize Targets Distribution
print('Targets Distribution Before SMOTE:', sorted(Counter(t).items()))

# OverSampling (OS) Fit and Transform the DataFrame
OS = SMOTE()
f, t = OS.fit_resample(f, t)

# Summarize the New Targets Distribution
print('Targets Distribution After SMOTE:', sorted(Counter(t).items()))
print("")

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv('FeaturesData.csv', index = False)
pd.DataFrame(t).to_csv('TargetData.csv', index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv('FeaturesData.csv')
df_t = pd.read_csv('TargetData.csv')

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv('MainDataFrame.csv', index = False)

print("------------------------------------------------------")
print("----------------- Data Understanding -----------------")
print("------------------------------------------------------")
print("")

print("Dataset Overview:")
print("*****************")
print(df.head(10))
print("")

print("General Information:")
print("********************")
print(df.info())
print("")

print("Statistics Information:")
print("***********************")
print(df.describe(include="all"))
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Samples Range:", df.index)
print("")

print(df.columns)
print("")

print("Missing Values (NaN):")
print("*********************")
print(df.isnull().sum())                                         
print("")

print("Duplicate Records:", df.duplicated().sum())
print("")   

print("Features Correlations:")
print("**********************")
print(df.corr(method='pearson'))
print("")

print("------------------------------------------------------")
print("--------------- Data Distribution --------------------")
print("------------------------------------------------------")
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Skewed Distribution of Features:")
print("********************************")
print(df.skew())
print("")
print(df.dtypes)
print("")

print("Target Distribution:")
print("********************")
print(df.groupby(df.iloc[:, -1].values).size())
print("")

print("------------------------------------------------------")
print("----------- Plotting Distribution of Data ------------")
print("------------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.hist(df)
plt.xlabel('Data Value', fontsize = 11)
plt.ylabel('Data Frequency', fontsize = 11)
plt.title('Data Distribution After Preparation')
plt.savefig('AutoDP_DataDistribution.png', dpi = 600)
plt.savefig('AutoDP_DataDistribution.tif', dpi = 600)
plt.show()
plt.close()

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")


