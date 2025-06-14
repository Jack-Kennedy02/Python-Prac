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

I ran one more statistical test to add value to the visuals I was being shown
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

### Results

![Visualization of Top Skills for Data Nerd](Project_Time/Images/skill_demand_percentages.png)

### Insights

- According with the statistcal test I ran, This result means that the relationship between age and body leangth is the hypothesis of age having a moderately postiive correlation with body length is statistically significant. As age increases so does body length.



## 2. Do different fur colors have an impact on the number of hours of sleep or sleep?

### Visualize Data

```
df_plot = df_DA_US_percent.iloc[:, :5]

sns.lineplot(data=df_plot, dashes=False, palette='tab10')
sns.set_theme(style='ticks')
sns.despine()

plt.title('Trending Top Skills for Data Analysts in the US')
plt.ylabel('Likelihood in Job Posting')
plt.xlabel('2023')
plt.legend().remove()


from matplotlib.ticker import PercentFormatter
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

for i in range(5):
    plt.text(11.2,df_plot.iloc[-1, i], df_plot.columns[i])

plt.show()
```

### Results

![Trending Top Skills for Data Analysts in the US](Project_Time/Images/Trending_Top_Skills.png)
*Bar graph visualizing the trending top skills for data analysts in the US in 2023.*

### Insights

-  SQL is the top skill for data analyst starting the year at 63% and dropping to 53% by the end of the year.

- Excel and Python saw some heavy fluctuations in the 2nd half of the year but saw the the most dramatic increases during the months of November and December. Excel increased its likelihood by 5.2% and Python increased by 5.3%.

- Power Bi and Tableau remained relatively stable throughout the 2023 with slight increases towards the end of the year.

## 3. How does the age of the cat affect its sleep?

### Salary Analysis for Data Nerds

```

```

#### Results

!['Salary Distributions of Data Jobs in the US](Project_Time/Images/Salary_Distribution_In_US.png)
*Box plot visualizing in the salary distributions for the top 6 data job titles.*

#### Insights

- Looking at the median salaries, the top job titles in salary distributions are the Senior Data Scientist and Data Engineer roles at around 130K USD. However, the entry and mid level title of Data Scientists and Enginers have shown outlier salaries that are significantly higher than their senior positions at 500K and 600K. Showing that an expertise in a very niche and desired skill can be compensated more without senior experience.

- The Senior Data Analyst roles recieves less compensation through the year than Data Scientists and Data engineers on the median. Making a increased incestive to jump from being a Data Analyst to either a Data Scientist or Data Engineer when making a new step in your career.

### Highest Paid & Most Demanded Skills for Data Analysts

### Visualize Data

```

```

### Results
In-demand skils for data analysts in the US:

![The Highest Paid & Most In-Demand SKills for Data Analysts in the US](Project_Time/Images/Most_In_Demand_Skills_For_Data_Analysts_In_The_US.png)
*Two separa bar graphs visualizing the highest paid skills and most in-demand skilsl for data analysts in the US.*

### Insights

- The gap between cloud softwares that are more specializaed compared to programming/vizualation/microsoft software ware is significant when it comes to compensation. The 10th highest cloud software is compensated around 150k  while the 10th highest non-cloud software is around 80k. It's safe to say that learning niche cloud programs will be compensated more, however, these skills may not be in demand as much as the lower paying non-cloud softwares.

- Programming and Visualation Softwares (python, tableau, sql) are on average more compesated then microsoft softwares (power bi, powerpoint, excel, word). Although this insight may not be as significant due to the gap being around 15K from top to bottom.

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
