import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
boston = pd.read_csv('Boston.csv')

# Print dataset summary information (optional)
pd.set_option('display.max_columns', None)
print("Feature names:", boston.columns.tolist())
print(boston.describe(), end="\n\n")
print("Total number of datapoints:", boston.shape[0], end="\n\n")

# Calculate Pearson correlation coefficient
corCoef, pValue = stats.pearsonr(boston['medv'], boston['lstat'])
print("Correlation coefficient between medv and lstat:", corCoef, ", with p-value:", pValue)

# Plot 1: Scatter plot with regression line, adding `hue` for the `black` column
sns.lmplot(
    data=boston,
    x='lstat',
    y='medv',
    hue='black',  # Color dots based on the 'black' column
    palette="Blues",  # Use a blue gradient similar to the original plot
    height=6,
    aspect=1.5,
    scatter_kws={'s': 50}  # Set dot size for better visibility
)
plt.xlabel("percent of households with low socioeconomic status")
plt.ylabel("median house value")
plt.title("Scatter Plot with Regression Line (Hue: Black)")
plt.show()

# Plot 2: Highlighting darker dots (lower `black` values)
# Define a threshold for darker dots (e.g., below median value of 'black')
black_threshold = boston['black'].median()
darker_dots = boston[boston['black'] < black_threshold]
lighter_dots = boston[boston['black'] >= black_threshold]

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(darker_dots['lstat'], darker_dots['medv'], c='black', label='Darker Dots (Low Black)', alpha=0.7, s=50)
plt.scatter(lighter_dots['lstat'], lighter_dots['medv'], c='lightblue', label='Lighter Dots (High Black)', alpha=0.7, s=50)
sns.regplot(data=boston, x='lstat', y='medv', scatter=False, color='blue')  # Regression line
plt.xlabel("percent of households with low socioeconomic status")
plt.ylabel("median house value")
plt.title("Scatter Plot with Regression Line (Highlighting Darker Dots)")
plt.legend()
plt.show()
