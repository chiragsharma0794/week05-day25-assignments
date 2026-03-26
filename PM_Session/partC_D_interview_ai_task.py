import pandas as pd

# Q1 - types of machine learning

print("Q1 - types of machine learning")
print("""
there are mainly 3 types:

1. supervised learning
   - the model is trained on labeled data
   - we give it inputs and the correct outputs
   - it learns the mapping between them
   - example: spam detection, house price prediction
   - sub types: regression (continuous output) and classification (category output)

2. unsupervised learning
   - no labels are given
   - the model finds patterns on its own
   - example: grouping customers into segments, topic modeling
   - common algorithms: kmeans, dbscan, pca

3. reinforcement learning
   - an agent learns by interacting with an environment
   - it gets rewards for good actions and penalties for bad ones
   - example: training a robot, game playing AI like alphago
   - no dataset needed, learns through trial and error

there is also semi supervised learning where you have a small amount
of labeled data and a large amount of unlabeled data but we mostly
focus on the 3 main types
""")


# Q2 - code to separate features and target

print("Q2 - separating features and target")

data = {
    'experience': [2, 5, 8, 1, 6, 3, 9, 4],
    'python_skill': [1, 1, 1, 0, 1, 0, 1, 1],
    'sql_skill': [0, 1, 1, 1, 0, 1, 1, 0],
    'education': [1, 2, 2, 0, 1, 1, 2, 1],
    'salary': [65000, 95000, 125000, 48000, 105000, 72000, 138000, 85000]
}

df = pd.DataFrame(data)
print("full dataset:")
print(df)

# separating features and target
X = df.drop(columns=['salary'])
y = df['salary']

print("\nX (features):")
print(X)

print("\ny (target):")
print(y.values)


# Q3 - regression vs classification

print("\nQ3 - difference between regression and classification")
print("""
regression:
- predicts a number (continuous value)
- like salary, price, temperature, age
- evaluation: mse, rmse, r2 score
- example algorithm: linear regression

classification:
- predicts a category or label
- like yes/no, spam/not spam, A/B/C grade
- evaluation: accuracy, precision, recall, f1 score
- example algorithm: logistic regression, decision tree

the easiest way to tell them apart:
look at the target column
if it has numbers like 95000 or 3.14 -> regression
if it has labels like yes, no, high, low -> classification
""")


# Part D - AI augmented task

print("\n\nPart D - AI Augmented Task")
print("""
prompt i used:
"explain types of machine learning with real world examples"

what the AI said:
- supervised learning: like a student learning from a teacher with answer keys
  example given was email spam detection and house price prediction
- unsupervised: like sorting objects without being told what they are
  example was customer segmentation
- reinforcement learning: like training a dog with treats
  example was game playing AI and self driving cars

is it correct?
yes everything was accurate and the analogies made it easy to understand.
the real world examples were good and relevant.

what i noticed:
- the AI also mentioned semi-supervised and self-supervised learning
  as modern extensions which was interesting
- the code examples used sklearn and were more advanced than what we 
  did in this assignment but they were correct
- the AI correctly explained that regression and classification are
  both under supervised learning which i found helpful for organizing
  my notes
""")
