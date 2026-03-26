import pandas as pd

# Part A - identifying types of ML problems

print("--- classifying the given problems ---")

print("""
1. A system predicts whether an email is spam or not using past labeled emails
   answer: supervised learning - classification
   reason: we have labeled data (spam / not spam) and output is a category

2. A retail company groups customers based on purchasing behavior without labels
   answer: unsupervised learning - clustering
   reason: no labels are given, the model has to find groups on its own
   this is typically done with kmeans or similar algorithms

3. A robot learns to walk by trying movements and getting rewards when correct
   answer: reinforcement learning
   reason: the robot is an agent that learns by getting rewards or penalties
   there is no dataset, it learns by doing

4. A model predicts house prices based on area, location, number of rooms
   answer: supervised learning - regression
   reason: we have labeled data and the target (price) is a continuous number

5. An ecommerce platform recommends similar products based on browsing history
   answer: unsupervised learning
   reason: no explicit labels, it finds patterns in user behavior
   can also be called collaborative filtering
""")

print("--- regression vs classification ---")
print("""
regression vs classification based on target variable:

dataset 1: predicting a students exam score
   -> regression, because score is a number (like 67 or 84)

dataset 2: predicting if a customer will buy a product (yes/no)
   -> classification, because output is a category

dataset 3: predicting number of job openings next month
   -> regression, continuous number

dataset 4: predicting which job title a person is suited for
   -> classification, output is one of several job titles

the main thing to check is: is the output a number or a category
if its a number -> regression
if its a label -> classification
""")

# creating a small dataframe for supervised ML
print("--- creating a supervised ML dataframe ---")

data = {
    'years_experience': [1, 3, 5, 7, 2, 8, 4, 6],
    'skills_count': [2, 3, 4, 4, 2, 5, 3, 4],
    'education': [0, 1, 1, 2, 0, 2, 1, 1],
    'is_remote': [1, 0, 1, 0, 1, 1, 0, 1],
    'salary': [55000, 72000, 90000, 115000, 60000, 130000, 80000, 105000]
}

df = pd.DataFrame(data)
print(df)

# separating features and target
X = df.drop('salary', axis=1)
y = df['salary']

print("\nfeatures X:")
print(X)

print("\ntarget y:")
print(y)

print("\nthis is a regression problem because salary is a continuous value")
