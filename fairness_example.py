# ai-ethics-and-fairness/fairness_example.py

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Train a logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)

# Example fairness check (disparate impact)
y_pred = model.predict(X_test)
print("Fairness metrics can be computed here.")
