import pandas as pd

df = pd.read_csv('AI_Job_Market_Trends_2026.csv')
df = df.drop_duplicates()

# filter 1 - senior people who know python and ml
print("--- senior ML engineers with python skill ---")
filtered1 = df[df['experience_level'] == 'Senior']
filtered1 = filtered1[filtered1['skills_python'] == 1]
filtered1 = filtered1[filtered1['skills_ml'] == 1]
print(filtered1[['job_title', 'country', 'salary', 'remote_type']].head(8))
print("count:", len(filtered1))

# filter 2 - remote jobs with high urgency and good salary
print("\n--- high urgency remote jobs above 150k ---")
filtered2 = df[(df['hiring_urgency'] == 'High') & (df['remote_type'] == 'Remote') & (df['salary'] > 150000)]
print(filtered2[['job_title', 'country', 'salary', 'experience_level']].head(8))
print("count:", len(filtered2))

# creating a new column for total skills a person has
print("\n--- new column: total_skills ---")
df['total_skills'] = df['skills_python'] + df['skills_sql'] + df['skills_ml'] + df['skills_deep_learning'] + df['skills_cloud']
print(df[['job_title', 'total_skills']].head(10))

# another new column - salary category
print("\n--- new column: salary_range ---")
def get_salary_range(sal):
    if sal < 80000:
        return 'low'
    elif sal < 150000:
        return 'medium'
    else:
        return 'high'

df['salary_range'] = df['salary'].apply(get_salary_range)
print(df['salary_range'].value_counts())

# sorting by salary
print("\n--- top 10 highest salaries ---")
sorted_df = df.sort_values('salary', ascending=False)
print(sorted_df[['job_title', 'country', 'salary', 'experience_level']].head(10))

# groupby - average salary by experience level
print("\n--- avg salary by experience level ---")
grp1 = df.groupby('experience_level')['salary'].mean()
print(grp1.sort_values(ascending=False))

# groupby - avg salary by job title
print("\n--- avg salary by job title ---")
grp2 = df.groupby('job_title')['salary'].mean()
print(grp2.sort_values(ascending=False))

# groupby - count of jobs per country
print("\n--- job count by country ---")
grp3 = df.groupby('country')['job_id'].count()
print(grp3.sort_values(ascending=False).head(10))

# groupby - remote type vs salary
print("\n--- avg salary by remote type ---")
grp4 = df.groupby('remote_type')['salary'].mean()
print(grp4)
