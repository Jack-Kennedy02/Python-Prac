# The Introduction
Welcome to my First Solo Project with out any help. I used the libraries of Pandas, Matplolib, Seaborn, Numpy, Scipy and Sklearn. Honestly, in the beginning of project I grabbeda  random
Tabular Data set that wasn't to complicated so I could conitnue my exploration into the libraries and eventually stumble into the sklearn libraries where you start building algorithms.
This project took me around 8 hours to complete on and off while working over a period of a couple of weeks. I'm extremely proud of myself for dedicating time to complete this project
on top of my normal life. Hopefully you readers enjoy what is beneath you!

# Background

Below are the questions I wanted to answer for this project:

1. What is the relationship beween age and body length?
2. Do different fur colors have an impact on the number of hours of sleep or sleep?
3. How does the age of the cat affect its sleep?

For the capstone of this project I decided to dive into to algorithms to see if I can make a sleep predictor based on various outputs.

# Tools Used

- Python: The backbone of my analysis, inconjuction with the follow libraries:
  - Pandas Library: This was used to analyze the data.
  - Matplotlib Library: Helped visualizing data.
  - Seaborn Library: Helped me create more advanced visuals.
  - Numpy: Helped me with mathematical computations
  - Scipy: Helped me with statistical tests.
  - Sklearn: Helped me with pre-built algorithms and classical machine learning.
- Jupyter Notebooks: The tool I used to run my Python scripts which let me easily include my notes and analysis.
- Visual Studio Code: My go-to for executing my python scripts
- Git & GitHub: Essential for version control and sharing my Python code and analysis.


# The Analysis

## 1. What is the relationship beween age and body length?

To investigate the relationship between age and body length I needed to gather data that wsa only relevant to the question I was answering, using the groupby function I grouped age with body length and weight (I included weight apart of the group by as well but didn't end up using it). I then began visualizing the graphs:

View my notebook with detailed steps here: [Visualizing_Kitties.ipynb](Python_Prac/Visualizing_Kitties.ipynb)

### Vizualize Data

``` 
sns.lineplot(data=age_summary, x='Age_in_years', y='Body_length')
sns.set_theme(style='ticks')

plt.title('Body_length vs. Age_in_years')
plt.xlabel('Age (Years)')
plt.ylabel('Body Length (cm)')

plt.show()
```
![V1 A](https://github.com/user-attachments/assets/719a9ad3-7b92-492d-b2b6-f3df40beb4f8)

I realized quickly that there was too much noise with my visual so I decided to use the .rolling() to solve my problem without sacraficing accuracy.

Adjusting my data input for the sns.lineplot.

```
age_summary['Body_length_smooth'] = age_summary['Body_length'].rolling(window=3, center=True).mean()

```
```
# Age vs. Body Length
sns.lineplot(data=age_summary, x='Age_in_years', y='Body_length_smooth')
sns.set_theme(style='ticks')

plt.title('Body_length vs. Age_in_years')
plt.xlabel('Age (Years)')
plt.ylabel('Body Length (cm)')

plt.show() 

```
![V1 B](https://github.com/user-attachments/assets/375be131-a36c-4427-837e-5e0bfdeb35b7)

*Line Plot visualizing the average Body Length as Age increases in a cat.*


At this point I saw a relationship was forming, however, to add validity to my findings I  decided 
to run one more statistical test, the Pearson Coefficient.

```
from scipy.stats import pearsonr

# Replace with your actual column names
x = clean_data['Age_in_years']
y = clean_data['Body_length_smooth']

# Perform Pearson correlation test
r_value, p_value = pearsonr(x, y)

# Print results
print(f"Correlation coefficient (r): {r_value:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value > 0.05:
    print("No statistically significant correlation (fail to reject H0)")
else:
    print("Statistically significant correlation (reject H0)")

```
Output:
```
Correlation coefficient (r): 0.599
P-value: 0.000
Statistically significant correlation (reject H0)
```

- According to pearson's correlation coefficient, the coefficient of 0.599 suggests a moderatle postive relationship between the two variables. Since the P-Value is 0 this means the moderatly postive relationship is statistically significant. This result means that the relationship between age and body leangth is the hypothesis of age having a moderately postiive correlation with body length is statistically significant. As age increases so does body length. However, there is a significant decrease in body length as the cat ages beyond the age of 9.


### Results

![V1 B](https://github.com/user-attachments/assets/375be131-a36c-4427-837e-5e0bfdeb35b7)

```
Correlation coefficient (r): 0.599
P-value: 0.000
Statistically significant correlation (reject H0)
```

### Insights

- As the cat ages, on average, the it begins to increase in size, in it's final years of life it seems to loss body length on average. This could be due to a number of factors like muscles losing mass or the lose of flexbility.

- The longest body length on average a cat in this sample had was 60 cm in height.


## 2. Do different fur colors have an impact on the number of hours of sleep or play?

To investigate this question I used value counts to understand which fur colors were signficant then
I built two bar graphs on two plots since sleep had a different unit of measure than play.

View my notebook with detailed steps here: [Visualizing_Kitties.ipynb](Python_Prac/Visualizing_Kitties.ipynb)

### Visualize Data

```
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Play_Time
breed_group['Play_Time'].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_ylabel('Average Play Time')
ax1.set_title('Average Play Time by Fur Colour')

# Plot Sleep
breed_group['Sleep'].plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_ylabel('Average Sleep Time')
ax2.set_title('Average Sleep Time by Fur Colour')

# X-axis label only on bottom plot since they share x-axis
ax2.set_xlabel('Fur Colour Dominant')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

### Results

![V2 A](https://github.com/user-attachments/assets/87182742-6284-47ad-b555-a49be9f7ae93)
*Two stacked Bar Charts visualizing the difference in average play and sleep time between fur colors*

### Insights

-  THere wasn't a huge difference in
-  The difference in averages for both play time and sleep time were not statistically signifcicant, meaning
   the observed difference had no significance relevance.
-  The fur color did not affect the average play and sleep time.

## 3. How does the age of the cat affect its sleep?

### Visualize Data

```
df['Age_Year'] = df['Age'].astype(int)
grouped = df.groupby('Age_Year')['Sleep'].mean().reset_index()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(data=grouped, x='Age_Year', y='Sleep')
plt.title('Average Sleep by Age (Year)')
plt.xlabel('Age (years)')
plt.ylabel('Average Sleep (hours)')
plt.show()
```

#### Results

![V3 A](https://github.com/user-attachments/assets/aacbd826-7088-4eba-ac23-0c269fc7c409)
*Bar Chart visualizing the average sleep for each year of the cat*

Running ANOVA Test:

```
from scipy import stats

# Create list of arrays, one per group
groups = [group['Sleep'].values for name, group in df.groupby('Age_Year')]

# Run ANOVA
f_stat, p_val = stats.f_oneway(*groups)
print(f"ANOVA test: F = {f_stat:.3f}, p = {p_val:.4f}")

```
```
ANOVA test: F = 65.226, p = 0.0000
```


#### Insights

- Cats sleep the most when they are young and older, being the most awake during their adult years.
- The high variance from the ANOVA test and P-Value suggest that the difference between the averages of the ages are statistcally significant further providing validity to the findings

## Sleep Predictor

Using linear regression we were found the R^2 value and the MSE (mean squared error. The MSE value is used to figure out how well a regression predicts values. 0 being a perfect model and large numbers indicate weaker models. 

### Testing Predictive Models

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop('Sleep_time_hours', axis=1)
X = pd.get_dummies(X)  # handle categorical variables
y = df['Sleep_time_hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
print("R²:", r2_score(y_test, preds))

```
```
results
```

```
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("R²:", r2_score(y_test, preds))
```

```
MSE: 3.1803871559633032
R²: 0.494688547433737
```
This results tell us the model is an average for the data and only around 50% of the data is explained by the variance with the R^2 value. We can do better.

```
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("R²:", r2_score(y_test, preds))

### Trying Gradient Boosting Regressor
```
```
MSE: 2.976999803827535
R²: 0.5270034680083053
```
Honestly the with the results of this predictor I could have went with this model, however, I wanted to try Catboost which is


### Choosing CatBoost


```
# Your dataframe: df
# Target variable:
y = df['Sleep_time_hours']
# Features:
X = df.drop('Sleep_time_hours', axis=1)

# List categorical feature column names or indices
categorical_features = [
    'Breed', 'Gender', 'Neutered_or_spayed', 'Fur_colour_dominant', 
    'Fur_pattern', 'Eye_colour', 'Allowed_outdoor', 'Preferred_food', 'Country'
]

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CatBoost Pool objects for train and test
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize and train model
model = CatBoostRegressor(iterations=121, learning_rate=0.1, depth=6, random_seed=42, verbose=100)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# Predict and evaluate
preds = model.predict(test_pool)

from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, preds))
print("R²:", r2_score(y_test, preds))
```
```
0:	learn: 2.4644598	test: 2.5530228	best: 2.5530228 (0)	total: 2.43ms	remaining: 2.43s
100:	learn: 1.3880068	test: 1.7616958	best: 1.7526340 (88)	total: 116ms	remaining: 1.03s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 1.750854189
bestIteration = 120

Shrink model to first 121 iterations.
MSE: 3.0654903896258303
R²: 0.5642077173021374
```
THe MSE value is a bit higher the nthe gradiant booster regressor however the R^2 value is 0.56 instead of 0.52, so this  is the best model we have to predict the data. 
Honestly it's pretty bang average, so I'm not genuinely impressed by my findings.


### Most imporant categories that affect the predictive model

```
# Get feature importance scores from your trained CatBoost model
importances = model.get_feature_importance(train_pool)

# Get feature names (make sure this matches the order of your training features)
feature_names = X_train.columns

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()
```

![V1_C](https://github.com/user-attachments/assets/2d824ff5-0fae-4466-a7d3-65cd0f0dc641)
*Yerrrrrrrrrrb*

### Insights

- Our best predictive model was the Catboost regressor and I wasn't too impressed by the R^2 and MSE scores, both indicated weak models that only explained maybe 56% of the data.
- The one category that immensily affected the facotr of sleep was age in years.

# What you learned

- Building Models: Designing models that created dummy variables to incorporate categorical columns in predictive models.

- Establishing Validity: Using statistcal formulas like pearson's coefficient, p-values, and ANOVA testing to help validate the correlations you potentially found.

- Establishing Good Data: Most of the categories in the data did not have an affect on sleep or any other feature. Building good datasets is the foundation of building
  better predictive models.


# Conclusion

Finding cool relationships like how body length and age, fur colors and play time, etc.. was a great experience to explore that any feature can have a relationships with another
feature. Another conclusion I reached was do my results actually matter or contribute to a somewhat useful questions with kitty data. I honestly wouldn't choose this data set
again to work with it's a good dataset to Extract and Transform but it was difficult to relate this to questions and answers for findings. It felt like the features were to
far apart from eachother. I'm very glad I did it though, it's a pain to write this Read Me's....
