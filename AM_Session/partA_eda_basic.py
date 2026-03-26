import pandas as pd

df = pd.read_csv('AI_Job_Market_Trends_2026.csv')

# first lets just see what the data looks like
print("first 5 rows")
print(df.head())

print("\nlast 5 rows")
print(df.tail())

# shape and columns
print("\nshape of dataset")
print(df.shape)

print("\ncolumn names")
print(df.columns)

# checking datatypes
print("\ndatatypes")
print(df.dtypes)

# now checking for missing values
print("\nmissing values in each column")
print(df.isnull().sum())

# looks like no missing values but lets double check
total_missing = df.isnull().sum().sum()
print("\ntotal missing values:", total_missing)

# also checking duplicates just to be safe
dupes = df.duplicated().sum()
print("duplicate rows:", dupes)

df = df.drop_duplicates()
print("shape after removing duplicates:", df.shape)

# descriptive statistics
print("\n--- describe() output ---")
print(df.describe())

# my observations from the stats
print("\nmy observations:")
print("- salary ranges from very low to over 300k, the mean is around 131k")
print("- years_experience goes from 0 to 30, average around 7-8 years")
print("- job_openings mean is around 5 which seems reasonable")
print("- skill columns are 0 or 1 so mean = % of jobs requiring that skill")
print("- python and ml seem required in roughly 55-60% of jobs")

# categorical columns
print("\n--- job title value counts ---")
print(df['job_title'].value_counts())

print("\n--- company size ---")
print(df['company_size'].value_counts())

print("\n--- experience level ---")
print(df['experience_level'].value_counts())

print("\n--- top 10 countries ---")
print(df['country'].value_counts().head(10))

print("\n--- remote type ---")
print(df['remote_type'].value_counts())

print("\n--- hiring urgency ---")
print(df['hiring_urgency'].value_counts())

print("\n--- education level ---")
print(df['education_level'].value_counts())
