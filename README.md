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



#### Insights

- 

- The Senior Data Analyst roles recieves less compensation through the year than Data Scientists and Data engineers on the median. Making a increased incestive to jump from being a Data Analyst to either a Data Scientist or Data Engineer when making a new step in your career.


## Sleep Predictor

### Visualize Data

```

```

### Results

![Most Optimal Skills for Data Analysts in the US](Project_Time/Images/Most_Optimal_Skills_for_Data_Analysts_in_the_US_with_Coloring_by_Technology.png)
*A scatter plot visualizing the most optimal skills (high paying & high demand) for data analysts in the US.*

### Insights

- The scatter plot showed that programming skills (colored in blue) are clustered around high median salaries than analyst_tools (colored in yellow). The range for programming tools are from 90k - 98k while the analyst tools are 82k to 93k. 

# What you learned

- Data Inconsistencies: Handling missing or inconsistent data entries requires careful consideration and thorough techniques to ensure the integrity  of the analysis. 

- Complex Data Visualization: Designing effect visual representation of complex datasets was challenging in conveying insights clrealy and compellingly.

- Balancing Breadth and Depth: Deciding how deeply to dive into each analysis while maintaining a broad overall landscape required constant balancing to ensure comprehensive coverage without getting lost in details.

# Insights

This project provided several insights into the data job market:

- Skill Demand and Salary Correlation: There is a clear correlation between the demand for specific skills and the salaries there skills command. Advanced and specialized skills like Python and Oracle often lead to higher Salaries.

- Market Trends: There are changing trends in skills demand, highlighting the dynamic nature of the data job market. Keeping up with these trends is essential for career growth in data analytics

- Economic Value of Skills: Understanding which skills are both in-demand and will-compensated can guide data danalysts in prioritizing learning to maximize their encomic returns.

# Conclusion

This exploration on the data job market has been incredibly informative, highlighting the critical skills needed to achieve the highest paying job titles. Gathering my own insights this project has personally influenced my career trajectory. Personally, I will be purusing python data projects, with the intentions to eventually pursue a masters' degree in data science. This project has also proved to be a valuable foundation for future opportunities and growth within the data job market and community.
