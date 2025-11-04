import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load and split data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# Log with MLflow
mlflow.log_metric("mse", mse)
mlflow.sklearn.log_model(model, "linear_model")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model trained successfully with MSE: {mse}")
