import pandas as pd # Never coded in R before but this seems to be the equivalent of library(pandas) in R

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

# load wage.csv
wage = pd.read_csv('wage.csv')

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# This will print '12' and not '11' because it counts the first 'empty' column
numFeatures = wage.shape[1]
print(numFeatures)

featureNames = wage.columns.tolist()
print(featureNames, end="\n\n")

# Drop the logwage column column
wage = wage.drop(columns=["logwage"])
numFeatures = wage.shape[1]
featureNames = wage.columns.tolist()
print("After dropping logwage", end="\n\n")
print(numFeatures)
print(featureNames)

# Will print 3000
numDataPoints = wage.shape[0]
print(numDataPoints, end="\n\n")

# display the data in the table
print(wage, end="\n\n")


# pd.set_option('display.max_columns', None) --> cant print all columns for whatever reason

print(wage.describe(), end="\n\n") # Similar to summary() in R

categorical_columns = ['maritl', 'education', 'jobclass', 'race']  # Add other columns as needed


# Step 1 (plots)

corCoef, pValue = stats.pearsonr(wage['wage'], wage['age'])

print(f"Pearson Correlation Coefficient: {corCoef:.2f}, P-value: {pValue:.2e}")

sns.lmplot(data=wage, x='age', y='wage', ci=95)

# Add labels to the plot
plt.xlabel("Age")
plt.ylabel("Wage")
plt.title(f"Scatter Plot with Regression Line\nCorrelation: {corCoef:.2f}")
plt.show() # Remember to make the window bigger to see the plot

# Step 2 (normality, shapiro, QQ plot): 

shapiroAge, pAge = stats.shapiro(wage['age'])
print(f"Shapiro-Wilk Test for 'age': Test Statistic = {shapiroAge:.4f}, P-value = {pAge:.4f}")

shapiroWage, pWage = stats.shapiro(wage['wage'])
print(f"Shapiro-Wilk Test for 'wage': Test Statistic = {shapiroWage:.4f}, P-value = {pWage:.4f}")

# Q-Q plot for "age"
sm.qqplot(wage['age'], line='s')
plt.title("Q-Q Plot for Age")
plt.ylabel("Age")
plt.show()

# Q-Q plot for "wage"
sm.qqplot(wage['wage'], line='s')
plt.title("Q-Q Plot for Wage")
plt.ylabel("Wage")
plt.show()

# Step 3 (pearsson correlation test):

# This is the exact same code as in step 1
corCoef, pValue = stats.pearsonr(wage['wage'], wage['age'])

print(f"Pearson Correlation Coefficient: {corCoef:.2f}, P-value: {pValue:.2e}, end='\n\n'")

# step 4

# Step 4.1 (list possible feature values):

# This is the same as levels(wage$education) in R
wage['education'] = wage['education'].astype('category')

levels = wage['education'].cat.categories
print("Levels:", levels.tolist())

# Step 4.2 (boxplot):

plt.figure(figsize=(10, 6))
sns.boxplot(data=wage, x='education', y='wage', palette='pastel', showmeans=True, meanline=True)
sns.stripplot(data=wage, x='education', y='wage', color='steelblue', jitter=0.21, size=5, alpha=0.7)

# Show the plot
plt.show()

# Step 4.3 (ANOVA):

# Perform one-way ANOVA
anova_model = sm.formula.ols('wage ~ C(education)', data=wage).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)

# have to compute mean squares manually
anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']

anova_table = anova_table[['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]

# Display the ANOVA table
print(anova_table)