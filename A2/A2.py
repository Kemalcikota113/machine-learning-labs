import pandas as pd # Never coded in R before but this seems to be the equivalent of library(pandas) in R

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

# load boston.csv
boston = pd.read_csv('Boston.csv')

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# This will print '15' and not '14' because it counts the first 'empty' column
numFeatures = boston.shape[1]
print(numFeatures)

featureNames = boston.columns.tolist()
print(featureNames, end="\n\n")

print(boston.describe(), end="\n\n")

print("total amount of datapoints: ", boston.shape[0], end="\n\n")

corCoef, pValue = stats.pearsonr(boston['medv'], boston['lstat'])

print("Correlation coefficient between medv and lstat: ", corCoef, ", with p-value: ", pValue)

sns.lmplot(data=boston, x='lstat', y='medv', hue="black", palette="Blues", height=6, aspect=2)

# Add labels to the plot
plt.xlabel("percent of households with low socioeconomic status")
plt.ylabel("median house value")
plt.title(f"Scatter Plot with Regression Line")
plt.show() # Remember to make the window bigger to see the plot