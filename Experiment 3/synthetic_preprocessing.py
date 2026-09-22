import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Create Synthetic Dataset
data = {
    "Age": [22, 25, None, 28, 30],
    "Salary": [25000, None, 35000, 40000, 45000],
    "Department": ["IT", "HR", "IT", None, "Sales"],
    "Years of Experience": [1, 2, 3, None, 5]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

# 2. Define Features
numeric_features = ["Age", "Salary", "Years of Experience"]
categorical_features = ["Department"]

# 3. Preprocess Numerical Data
num_imputer = SimpleImputer(strategy="median")
df[numeric_features] = num_imputer.fit_transform(df[numeric_features])

# 4. Preprocess Categorical Data
cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

# 5. One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
department_encoded = encoder.fit_transform(df[["Department"]])

encoded_df = pd.DataFrame(
    department_encoded,
    columns=encoder.get_feature_names_out(["Department"])
)

# 6. Combine Data
df_final = pd.concat(
    [df[numeric_features].reset_index(drop=True), encoded_df],
    axis=1
)

print("\n--- Preprocessed Dataset ---")
print(df_final)