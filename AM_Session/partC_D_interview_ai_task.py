import pandas as pd

df = pd.read_csv('AI_Job_Market_Trends_2026.csv')

# Q1 - what is EDA and why is it important

print("Q1 - What is EDA and why is it important")
print("""
EDA stands for Exploratory Data Analysis. it basically means looking at 
your data carefully before doing anything else like building models.

you check things like:
- how many rows and columns are there
- are there any missing values
- what do the numbers look like (mean, max, min etc)
- are there any weird or extreme values (outliers)
- what are the categories in text columns

why is it important:
- if you skip EDA and directly train a model the results will probably
  be wrong because your data might have issues you didnt know about
- it helps you understand what columns are useful and which ones arent
- you can spot if a column has wrong data types (like salary stored as text)
- you get a feel for the data before writing complex code
- basically its like reading the instructions before assembling something
""")


# Q2 - filter rows where salary is greater than average

print("Q2 - filter rows where salary > average")

avg_salary = df['salary'].mean()
print("average salary:", avg_salary)

above_avg = df[df['salary'] > avg_salary]
print("rows above average salary:", len(above_avg))
print(above_avg[['job_title', 'country', 'salary']].head(8))


# Q3 - insights from describe()

print("\nQ3 - what can we get from describe()")
print(df.describe())
print("""
from describe() we can see:
- count tells us how many non-null values there are in each column
  if count is less than total rows then there are missing values
- mean is the average value
- std shows how spread out the values are, high std means more variation
- min and max show the range of values
- 25%, 50%, 75% are quartiles
  50% is the median which is useful when data is skewed

for example in this dataset:
- salary mean is around 131k but std is high (around 50k) 
  so salaries vary a lot depending on country and job role
- years_experience max is 30 which makes sense for senior folks
- job_openings mean is around 5 per posting
""")


# Part D - AI augmented task

print("\n\nPart D - AI Augmented Task")
print("""
prompt i used:
"Explain EDA steps using Pandas with examples."

what the AI explained:
the AI gave a step by step EDA process:
1. load data with read_csv
2. use head() and tail() to see the data
3. check shape and dtypes
4. use isnull().sum() to find missing values
5. use fillna() or dropna() to handle missing values
6. use describe() for statistics
7. use value_counts() for categorical columns
8. use groupby for aggregated analysis

is it correct?
yes the steps are correct and make sense. the code examples were clean
and easy to follow. the AI used a sample dataset about house prices.

one thing i noticed - the AI didnt mention checking for duplicate rows
which is also an important step in data cleaning. i added that in my code.
""")
