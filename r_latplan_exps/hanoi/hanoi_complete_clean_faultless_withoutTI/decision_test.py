import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create dummy data
np.random.seed(42)
n_samples = 1000
n_features = 10
X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, 2, size=n_samples)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LGBMClassifier
lgbm_clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.1, n_estimators=100)


# Train the model with callbacks
lgbm_clf.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    callbacks=[
        early_stopping(stopping_rounds=10),  # Stops training if no improvement
        log_evaluation(10),  # Logs evaluation every 10 rounds
    ]
)

# Make predictions
y_pred = lgbm_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
