<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Kaggle Competitions</h1>
    <p>Welcome to my repository for Kaggle competition entries! Here, you'll find various projects where I tackle different machine learning challenges. This repository will document my journey through these competitions, showcasing my approaches, models, and improvements over time.</p>

  <h2>Projects</h2>

  <h3>Titanic: Machine Learning from Disaster</h3>
    <h4>Description</h4>
    <p>This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. I used a Random Forest Classifier to make predictions based on passenger data.</p>

  <h4>Code</h4>
    <pre>
<code>
# Import necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# Load and display input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load training and test data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# Display data
print(train_data.head())
print(test_data.head())

# Calculate survival rates by gender
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("% of men who survived:", rate_men)

# Prepare data for modeling
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Save predictions to CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
</code>
</pre>

  <h4>Next Steps</h4>
    <ul>
        <li><strong>Hyperparameter Tuning:</strong> Experiment with different parameters for the Random Forest model.</li>
        <li><strong>Feature Engineering:</strong> Create new features like Family Size and Is Alone.</li>
        <li><strong>Cross-Validation:</strong> Validate the model's performance using cross-validation.</li>
    </ul>

  <p>Stay tuned for more updates and projects as I continue to explore and learn from Kaggle competitions!</p>
</body>
</html>
