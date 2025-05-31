import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset.csv assuming it is in the project root or appropriate folder
data = pd.read_csv('dataset.csv')

# Inspect your data to understand features and target
print(data.head())

# Suppose the last column is the target
X = data.iloc[:, :-1].values  # all rows, all columns except last
y = data.iloc[:, -1].values   # target column

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/iris_model.pkl')
