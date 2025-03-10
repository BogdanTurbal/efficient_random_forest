#!/usr/bin/env python3
"""
This script loads the Adult dataset from OpenML, one-hot encodes categorical features,
and writes two files ("train.txt" and "test.txt") that follow the format expected by the
C++ code:
   - Training file: each line begins with the label (0 or 1) followed by space‐separated tokens
     of the form "feature_index:feature_value".
   - Test file: same format but without the label.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ---------------- Load and Prepare the Adult Dataset ----------------

# Fetch the Adult dataset from OpenML (version=2 returns a DataFrame)
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame
print(df)

# The target column is named "class" and contains values like ">50K" or "<=50K"
# Map them to integers: 1 for income >50K and 0 otherwise.
df['class'] = df['class'].apply(lambda x: 1 if x.strip() in [">50K", ">50K."] else 0)

# Identify numeric and categorical columns.
# Numeric columns: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'class' in numeric_cols:
    numeric_cols.remove('class')

# Categorical columns: all object columns (e.g. workclass, education, marital-status, etc.)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# One-hot encode categorical features.
df_cat = pd.get_dummies(df[categorical_cols], drop_first=False)

# Concatenate numeric features, one-hot encoded categorical features, and the target.
df_final = pd.concat([df[numeric_cols], df_cat, df['class']], axis=1)
print("Feature size:", df_final.shape[1] - 1)

# To ensure a consistent order of features, sort all feature column names.
feature_cols = sorted(list(df_final.columns))
feature_cols.remove('class')
# Reorder columns so that feature columns come first and target ("class") is last.
df_final = df_final[feature_cols + ['class']]

# Split into training and testing sets.
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)

# ---------------- Write Data Files in the Required Format ----------------

def write_data(df, filename, is_train=True):
    """
    Write the DataFrame to a text file in sparse format.
    For training data, each line begins with the label followed by space-separated tokens "i:value"
    for each nonzero feature. For test data, the label is omitted.
    """
    # The feature columns (order is fixed)
    feat_cols = list(df.columns)
    feat_cols.remove('class')
    with open(filename, 'w') as f:
        for idx, row in df.iterrows():
            if is_train:
                # First token is the label (as an integer)
                line = f"{int(row['class'])}"
            else:
                line = ""
            # For each feature, output "index:value" if nonzero.
            for i, col in enumerate(feat_cols):
                val = row[col]
                # Only include nonzero values to produce a sparse format.
                if val != 0:
                    line += f" {i}:{val:.6f}"
            f.write(line.strip() + "\n")

# Write training file (includes labels).
write_data(train_df, "./train.txt", is_train=True)
# Write test file (without labels).
write_data(test_df, "./test.txt", is_train=True)

print("Train and test files have been written.")
