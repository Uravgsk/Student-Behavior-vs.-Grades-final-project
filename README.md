# Student-Behavior-vs.-Grades-final-project

# Task
You should finish at least dataset preparation, implement at 1 existing method, and evaluation.
• Here are some examples
1. Dataset :
1) Data visualization
2) Data preprocessing
3) Date loading
2. 1 Existing Method (10 Points).
1) Explain the algorithm you used.
2) Show the preliminary evaluation results.
3) Analysis of the results and any modifications or plans before the final report deadline.
3. Model Evaluation


Here is all the data you need:
"Students_Grading_Dataset.csv"

## Data loading

### Subtask:
Load the "Students_Grading_Dataset.csv" file into a pandas DataFrame.


**Reasoning**:
Load the dataset and display its first few rows and shape to verify the loading.



#Written by Ayomide
import pandas as pd

try:
    df = pd.read_csv('Students_Grading_Dataset.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'Students_Grading_Dataset.csv' not found.")
    df = None # Assign None to df in case of error
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None # Assign None to df in case of error

## Data exploration

### Subtask:
Explore the dataset to understand its structure, identify missing values, data types of each column, and the distribution of key variables like grades, attendance, and study hours. Determine the shape of the data and look for potential outliers.


**Reasoning**:
Explore the dataset by examining its structure, identifying missing values, analyzing key variable distributions, checking for outliers, and determining the data shape.



#Written by Ayomide
import matplotlib.pyplot as plt
import seaborn as sns

# Examine Data Structure
display(df.info())
display(df.describe())

# Identify Missing Values
display(df.isnull().sum())

# Analyze Key Variable Distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['Grade'], kde=True)
plt.title('Distribution of Grades')

plt.subplot(1, 3, 2)
sns.histplot(df['Attendance (%)'], kde=True)
plt.title('Distribution of Attendance (%)')

plt.subplot(1, 3, 3)
sns.histplot(df['Study_Hours_per_Week'], kde=True)
plt.title('Distribution of Study Hours per Week')

plt.tight_layout()
plt.show()

# Check for Outliers in numerical features
numerical_features = ['Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg',
                      'Participation_Score', 'Projects_Score', 'Total_Score',
                      'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']

plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_features):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Determine Data Shape
print("Shape of the DataFrame:", df.shape)


## Data cleaning

### Subtask:
Clean the data by handling missing values and outliers.


**Reasoning**:
Fill missing values in 'Parent_Education_Level' with the mode, handle outliers in numerical features using IQR, remove duplicates, and visualize the specified columns' distributions.



#Written by Ayomide

# Fill missing values in 'Parent_Education_Level' with the mode
parent_education_mode = df['Parent_Education_Level'].mode()[0]
df['Parent_Education_Level'].fillna(parent_education_mode, inplace=True)

# Handle outliers using IQR method
numerical_features = ['Midterm_Score', 'Final_Score', 'Total_Score', 'Study_Hours_per_Week']
for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Remove duplicate rows
num_duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
num_duplicates_after = df.duplicated().sum()

print(f"Number of duplicate rows before removal: {num_duplicates_before}")
print(f"Number of duplicate rows after removal: {num_duplicates_after}")


# Visualize distributions after outlier handling
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['Grade'], kde=True, color='skyblue')
plt.title('Distribution of Grades')
plt.subplot(1, 3, 2)
sns.histplot(df['Attendance (%)'], kde=True, color='salmon')
plt.title('Distribution of Attendance')
plt.subplot(1, 3, 3)
sns.histplot(df['Study_Hours_per_Week'], kde=True, color='lightgreen')
plt.title('Distribution of Study Hours')
plt.tight_layout()
plt.show()

**Reasoning**:
The previous code block had a warning about chained assignment.  I will fix this and re-run the code.  The subtask is completed after this.



#Written by Ayomide

# Fill missing values in 'Parent_Education_Level' with the mode
parent_education_mode = df['Parent_Education_Level'].mode()[0]
df['Parent_Education_Level'] = df['Parent_Education_Level'].fillna(parent_education_mode)

# Handle outliers using IQR method
numerical_features = ['Midterm_Score', 'Final_Score', 'Total_Score', 'Study_Hours_per_Week']
for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Remove duplicate rows
num_duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
num_duplicates_after = df.duplicated().sum()

print(f"Number of duplicate rows before removal: {num_duplicates_before}")
print(f"Number of duplicate rows after removal: {num_duplicates_after}")


# Visualize distributions after outlier handling
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['Grade'], kde=True, color='skyblue')
plt.title('Distribution of Grades')
plt.subplot(1, 3, 2)
sns.histplot(df['Attendance (%)'], kde=True, color='salmon')
plt.title('Distribution of Attendance')
plt.subplot(1, 3, 3)
sns.histplot(df['Study_Hours_per_Week'], kde=True, color='lightgreen')
plt.title('Distribution of Study Hours')
plt.tight_layout()
plt.show()

## Data preparation

### Subtask:
Prepare the data for model training by converting categorical features to numerical representations and scaling numerical features.  Then, split the data into training and testing sets.


**Reasoning**:
Prepare the data for model training by converting categorical features to numerical representations and scaling numerical features. Then, split the data into training and testing sets.



#Written by Ayomide

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('Students_Grading_Dataset.csv')

# One-hot encode categorical features
categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Grade']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# Feature scaling for numerical features
numerical_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Drop Student_ID, First_Name, Last_Name, Email
df = df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1)

# Split data into training and testing sets
X = df.drop(['Grade_A', 'Grade_B', 'Grade_C', 'Grade_D', 'Grade_F'], axis=1)
y = df[['Grade_A', 'Grade_B', 'Grade_C', 'Grade_D', 'Grade_F']]
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display(df_train.head())
display(df_test.head())

**Reasoning**:
The previous code failed due to a KeyError, indicating that the column 'Grade_E' was not found.  I will re-examine the code to ensure that 'Grade_E' is correctly handled during one-hot encoding and subsequent data splitting.  I will generate a corrected code block.



#Written by Isiah
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('Students_Grading_Dataset.csv')

# One-hot encode categorical features
categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Grade']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, encoded_df], axis=1)


# Feature scaling for numerical features
numerical_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Drop Student_ID, First_Name, Last_Name, Email
df = df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1)

# Split data into training and testing sets
X = df.drop(list(df.filter(regex='Grade_')), axis=1)
y = df.filter(regex='Grade_')
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display(df_train.head())
display(df_test.head())

**Reasoning**:
The error indicates that the categorical columns specified in `categorical_cols` are no longer present in the DataFrame `df`.  This likely occurred because the data cleaning steps modified or removed those columns. I will review the data cleaning steps and correct the code to ensure the correct columns are used for one-hot encoding. I will then re-execute the entire data preparation process.



#Written by Isiah
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('Students_Grading_Dataset.csv')

# One-hot encode categorical features
categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Grade']
# Check if all categorical columns exist in the dataframe
missing_cols = set(categorical_cols) - set(df.columns)
if missing_cols:
    print(f"Warning: The following categorical columns are missing: {missing_cols}")
    # Handle the missing columns appropriately (e.g., remove from categorical_cols or fill with default values)
    categorical_cols = list(set(categorical_cols) - missing_cols)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# Feature scaling for numerical features
numerical_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Drop Student_ID, First_Name, Last_Name, Email
df = df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1, errors='ignore')

# Split data into training and testing sets
X = df.drop(list(df.filter(regex='Grade_')), axis=1, errors='ignore')
y = df.filter(regex='Grade_')
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display(df_train.head())
display(df_test.head())

## Data visualization

### Subtask:
Visualize the relationships between features and the target variable (Grade) and the distribution of data after cleaning and preprocessing.


**Reasoning**:
Visualize the relationships between features and the target variable (Grade) and the distribution of data after cleaning and preprocessing using histograms, box plots, and scatter plots.



#Written by Isiah
import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for key numerical features
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
sns.histplot(df_train['Study_Hours_per_Week'], kde=True, color='skyblue')
plt.title('Distribution of Study Hours per Week')
plt.subplot(1, 4, 2)
sns.histplot(df_train['Attendance (%)'], kde=True, color='salmon')
plt.title('Distribution of Attendance')
plt.subplot(1, 4, 3)
sns.histplot(df_train['Midterm_Score'], kde=True, color='lightgreen')
plt.title('Distribution of Midterm Scores')
plt.subplot(1, 4, 4)
sns.histplot(df_train['Final_Score'], kde=True, color='orange')
plt.title('Distribution of Final Scores')
plt.tight_layout()
plt.show()


# Box plots for key numerical features
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
sns.boxplot(y=df_train['Study_Hours_per_Week'], color='skyblue')
plt.title('Box Plot of Study Hours per Week')
plt.subplot(1, 4, 2)
sns.boxplot(y=df_train['Attendance (%)'], color='salmon')
plt.title('Box Plot of Attendance')
plt.subplot(1, 4, 3)
sns.boxplot(y=df_train['Midterm_Score'], color='lightgreen')
plt.title('Box Plot of Midterm Scores')
plt.subplot(1, 4, 4)
sns.boxplot(y=df_train['Final_Score'], color='orange')
plt.title('Box Plot of Final Scores')
plt.tight_layout()
plt.show()


# Scatter plots for numerical features vs. Grade (example with Grade_A)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='Study_Hours_per_Week', y=y_train['Grade_A'], data=df_train, hue=y_train['Grade_A'], palette='viridis')
plt.title('Study Hours vs. Grade A')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Attendance (%)', y=y_train['Grade_A'], data=df_train, hue=y_train['Grade_A'], palette='viridis')
plt.title('Attendance vs. Grade A')

plt.tight_layout()
plt.show()

## Model training

### Subtask:
Train a Linear Regression model to predict student grades based on the preprocessed features.


**Reasoning**:
Train a Linear Regression model and make predictions.



#Written by Isiah
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(df_train, y_train)

# Make predictions
y_pred = model.predict(df_test)

# Evaluate the model (optional, but recommended)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print model coefficients and intercept (optional)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# A brief explanation of the linear regression algorithm:
# Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables.
# It assumes a linear relationship, meaning the change in the dependent variable is proportional to the change in the independent variables.
# The model aims to find the best-fitting line (or hyperplane in multiple dimensions) that minimizes the sum of squared errors between the predicted and actual values.

## Model evaluation

### Subtask:
Evaluate the performance of the trained Linear Regression model.


**Reasoning**:
Evaluate the model's performance using MSE, RMSE, R-squared, and MAE. Analyze the results and suggest next steps.



#Written by Isiah
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Predict the target variables on the test set
y_pred = model.predict(df_test)

# Evaluate the model for each grade
for i, col in enumerate(y_test.columns):
    mse = mean_squared_error(y_test[col], y_pred[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test[col], y_pred[:, i])
    mae = mean_absolute_error(y_test[col], y_pred[:,i])

    print(f"Evaluation for Grade {col}:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print("-" * 20)

# Analysis of the results:
# Examine the R-squared and error metrics for each grade. Low R-squared values and high error metrics indicate poor model fit.
# Consider the following potential reasons for poor performance:
# 1. Data preprocessing issues: Check for missing values, outliers, or incorrect scaling.
# 2. Feature engineering: Explore additional features or transformations that might improve the model.
# 3. Model suitability: Consider other regression models such as polynomial regression or ridge/lasso regression.
# 4. Multicollinearity: Check for high correlation among features.
# 5. Insufficient data: Collect more data if possible.

# Potential next steps:
# 1. Feature engineering: try creating interaction terms or polynomial terms
# 2. Model selection: explore different regression models like Ridge or Lasso regression
# 3. Hyperparameter tuning: optimize the regularization parameter (alpha) for Ridge or Lasso regression
# 4. Data exploration: visualize relationships between features and target to guide feature engineering

## Model optimization

### Subtask:
Optimize the linear regression model by exploring alternative regression models (Ridge and Lasso) and performing hyperparameter tuning.


**Reasoning**:
Import necessary libraries and perform hyperparameter tuning for Ridge and Lasso regression models using GridSearchCV. Evaluate the best models on the test data and compare their performance with the initial linear regression model.



#Written by Isiah
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Define parameter grids for Ridge and Lasso
alphas = np.logspace(-3, 3, 7)
param_grid_ridge = {'alpha': alphas}
param_grid_lasso = {'alpha': alphas}

# Create Ridge and Lasso regression models
ridge_model = Ridge()
lasso_model = Lasso()

# Use GridSearchCV to find the best hyperparameters
ridge_grid = GridSearchCV(ridge_model, param_grid_ridge, scoring='neg_mean_squared_error', cv=5)
lasso_grid = GridSearchCV(lasso_model, param_grid_lasso, scoring='neg_mean_squared_error', cv=5)

# Fit GridSearchCV objects to the training data
ridge_grid.fit(df_train, y_train)
lasso_grid.fit(df_train, y_train)

# Evaluate the best models on the test data
y_pred_ridge = ridge_grid.predict(df_test)
y_pred_lasso = lasso_grid.predict(df_test)

# Evaluate the models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")

evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
evaluate_model(y_test, y_pred_lasso, "Lasso Regression")
print("Linear Regression Performance (from previous step):") # use the linear regression output from previous step.
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Analyze results, identify best model, and propose further improvements.
# This part will be done in the next step.

**Reasoning**:
Analyze the results of the optimized models, compare their performance with the original linear regression, discuss the impact of regularization, identify the best-performing model, and propose further improvements.



#Written by Isiah
# Analyze the results
print("Analysis of Results:")
print("Ridge Regression, Lasso Regression, and Linear Regression models were evaluated using MSE, RMSE, R-squared and MAE.")
print("All three models show poor performance with negative R-squared values, indicating that they do not fit the data well.")
print("Ridge and Lasso regression did not significantly improve the model performance compared to the original linear regression.")

# Identify the best-performing model
print("\nBest Performing Model:")
print("Based on the evaluation metrics, none of the models perform well.  All three models (Linear Regression, Ridge Regression, and Lasso Regression) have very similar performance, all with negative R-squared scores and high error values.")

# Propose further improvements
print("\nFurther Improvements:")
print("1. Feature Engineering: Explore creating interaction terms, polynomial features, or other transformations of the existing features.  Investigate removing irrelevant or redundant features.")
print("2. Model Selection:  Consider trying other regression models, such as Support Vector Regression (SVR), Decision Tree Regression, or Random Forest Regression.")
print("3. Data Exploration: Conduct a more in-depth analysis of feature distributions, correlations, and potential outliers. Consider visualizing the relationships between features and the target variables.")
print("4. Regularization parameter tuning: Although tuning did not improve the model much, try exploring a wider range of alpha values or consider other regularization techniques.")
print("5. Address Class Imbalance: Explore the distribution of the target variables (grades).  If there's a significant imbalance in the distribution of grades, techniques like oversampling or undersampling or cost-sensitive learning may be helpful. ")
print("6. Data Collection: More data may help to increase the reliability of the model.")

print("\nFurther Steps:")
print("Before the final report deadline, focus on feature engineering and exploring other regression models.  Evaluate their performance with robust cross-validation and carefully analyze the results.")

## Summary:

### 1. Q&A

* **What is the performance of the trained Linear Regression model?** The initial Linear Regression model performed poorly, with a negative R-squared value and a relatively high Mean Squared Error (MSE).  This indicates the model does not fit the data well and cannot accurately predict student grades.  Subsequent attempts using Ridge and Lasso regression with hyperparameter tuning did not significantly improve performance, and all models show similar poor performance with negative R-squared scores and high error values.

* **What are the next steps to improve the model's performance?**  Further improvements include feature engineering (creating interaction terms, polynomial features), exploring alternative regression models (SVR, Decision Tree Regression, Random Forest Regression), in-depth data exploration, refining regularization parameter tuning, addressing potential class imbalance, collecting more data.  The next immediate steps should focus on feature engineering and exploring other regression models.

### 2. Data Analysis Key Findings

* **Missing Data:** 1025 missing values were found in the 'Parent_Education_Level' column, which were imputed using the mode of the column.
* **Outlier Handling:** Outliers in numerical features ('Midterm_Score', 'Final_Score', 'Total_Score', 'Study_Hours_per_Week') were handled using the IQR method.
* **Data Preparation:**  Categorical features were one-hot encoded, numerical features were standardized, and the data was split into training and testing sets (80/20 split).  A warning was generated regarding missing categorical columns, which were subsequently removed from the encoding process.
* **Model Performance:** All tested models (Linear Regression, Ridge Regression, Lasso Regression) showed poor performance, indicated by negative R-squared values and high error metrics (MSE, RMSE, MAE).  This suggests the models do not effectively capture the relationships between the features and the target variables.

### 3. Insights or Next Steps

* **Prioritize Feature Engineering and Model Selection:**  The current features and the linear model assumptions appear inadequate.  Focus on creating new features, exploring transformations of existing features and evaluating different regression model types (e.g., tree-based models, SVR) to find a better fit for the data.
* **Investigate Class Imbalance:** Analyze the distribution of grades to check for class imbalance.  If present, techniques like oversampling, undersampling, or cost-sensitive learning might improve the model's performance on under-represented grades.


# Logistic Regression Model


#Written by Ayomide
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('Students_Grading_Dataset.csv')

# Drop unnecessary columns
X = df.drop(['Grade', 'Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1)
y = df['Grade']

# One-hot encode the target variable
encoder_y = OneHotEncoder(sparse_output=False)
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Encode categorical features
categorical_features = ['Gender', 'Department', 'Extracurricular_Activities',
                        'Internet_Access_at_Home', 'Parent_Education_Level',
                        'Family_Income_Level']
encoder_X = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_X.fit(X_train[categorical_features])

# Transform categorical features
X_train_encoded = pd.DataFrame(encoder_X.transform(X_train[categorical_features]),
                               columns=encoder_X.get_feature_names_out(categorical_features),
                               index=X_train.index)
X_test_encoded = pd.DataFrame(encoder_X.transform(X_test[categorical_features]),
                              columns=encoder_X.get_feature_names_out(categorical_features),
                              index=X_test.index)

# Replace categorical features with encoded features
X_train = X_train.drop(columns=categorical_features).join(X_train_encoded)
X_test = X_test.drop(columns=categorical_features).join(X_test_encoded)

# Scale numerical features
numerical_features = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score',
                      'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score',
                      'Projects_Score', 'Total_Score', 'Study_Hours_per_Week',
                      'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train Logistic Regression model
logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logistic_model.fit(X_train, y_train.argmax(axis=1))  # Assuming one-hot encoding

# Make predictions
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate the model
evaluate_model(y_test.argmax(axis=1), y_pred_logistic, "Logistic Regression")
