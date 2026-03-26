import pandas as pd

# Part B - regression and classification examples

print("--- regression example ---")
print("""
use case: predicting salary of an AI professional

input (features):
- years of experience
- number of skills they have
- education level (bachelor=0, master=1, phd=2)
- whether remote or not (1/0)

output:
- salary in USD (a number like 95000 or 142000)

why regression:
- salary is a continuous number, not a category
- the model needs to predict any real number not just yes or no
""")

# creating sample data for regression
reg_data = {
    'years_exp': [1, 2, 4, 5, 7, 9, 3, 6, 8, 10],
    'skills': [2, 2, 3, 4, 4, 5, 3, 4, 5, 5],
    'salary': [55000, 62000, 78000, 92000, 108000, 140000, 70000, 100000, 125000, 148000]
}

df_reg = pd.DataFrame(reg_data)
print("regression dataset:")
print(df_reg)

# simple prediction using a formula i came up with
# salary = years * 9000 + skills * 7000 + 28000
df_reg['predicted_salary'] = df_reg['years_exp'] * 9000 + df_reg['skills'] * 7000 + 28000
print("\nwith predictions:")
print(df_reg)

# calculating error manually
df_reg['error'] = df_reg['salary'] - df_reg['predicted_salary']
mse = (df_reg['error'] ** 2).mean()
print("\nmse:", round(mse, 2))
print("rmse:", round(mse ** 0.5, 2))


print("\n--- classification example ---")
print("""
use case: predicting if a job has high hiring urgency or not

input (features):
- number of job openings
- salary offered
- number of required skills

output:
- hiring_urgency: high or low (a category)

why classification:
- the output is a label from a fixed set (high/low)
- we dont need an exact number, just which group it belongs to
""")

cls_data = {
    'openings': [2, 9, 5, 10, 1, 8, 3, 7],
    'salary': [75000, 165000, 105000, 180000, 68000, 155000, 88000, 148000],
    'skills_needed': [2, 5, 3, 5, 2, 4, 3, 4],
    'urgency': ['low', 'high', 'high', 'high', 'low', 'high', 'low', 'high']
}

df_cls = pd.DataFrame(cls_data)
print("classification dataset:")
print(df_cls)

# simple rule - if openings > 6 or salary > 140000 then high urgency
def predict_urgency(row):
    if row['openings'] > 6 or row['salary'] > 140000:
        return 'high'
    else:
        return 'low'

df_cls['predicted'] = df_cls.apply(predict_urgency, axis=1)
print("\nwith predictions:")
print(df_cls)

correct = (df_cls['urgency'] == df_cls['predicted']).sum()
acc = correct / len(df_cls) * 100
print(f"\naccuracy: {correct}/{len(df_cls)} = {acc}%")
