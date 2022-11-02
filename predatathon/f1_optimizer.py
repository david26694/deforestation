# %%
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Generate multiclass data with sklearn
N_CLASSES = 8

# %%
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10, n_classes=N_CLASSES,
    weights=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025],
    random_state=0)

# %%
# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Create a model
model = HistGradientBoostingClassifier(random_state=42)
# %%
# Fit the model
model.fit(X_train, y_train)
# %%
# Predict on test data
y_pred = model.predict_proba(X_test)
# %%
# Calculate f1 score
f1_score(y_test, y_pred.argmax(axis=1), average="macro")
# %%
f1_score(y_test, y_pred.argmax(axis=1), average="micro")
# %%
pd.Series(y).value_counts()
# %%
def decision_fn(x: ArrayLike, p: ArrayLike):
    # Sort x by p
    indices = np.argsort(x)[::-1]
    x_sorted = x[indices]
    p_sorted = p[indices]
    # Find the first index where x > p
    if (x_sorted > p_sorted).any():
        return indices[(x_sorted > p_sorted)][0]
    else:
        # Return max index
        return np.argmax(x)

# %%
x = np.array([0.1, 0.5, 0.3, 0.4])
p = np.array([1] * 4)

decision_fn(x, p)

# %%
x = np.array([0.1, 0.2, 0.3, 0.4])
p = np.array([0.2, 0.1, 0.4, 0.3])

decision_fn(x, p)
# %%
x = np.array([0.1, 0.2, 0.3, 0.3])
p = np.array([0.0, 0.1, 0.4, 0.3])

decision_fn(x, p)

# %%
x = np.array([0.2, 0.1, 0.3, 0.3])
p = np.array([0.0, 0.1, 0.4, 0.3])

decision_fn(x, p)

# %%
x = np.array([0.1, 0.1, 0.2, 0.2])
p = np.array([0.0, 0.1, 0.4, 0.3])

decision_fn(x, p)

# %%
x = np.array([0.1, 0.1, 0.5, 0.2])
p = np.array([0.0, 0.1, 0.4, 0.3])

decision_fn(x, p)

# %%
def scoring_fn(y_test, y_proba, p):
    y_pred = np.apply_along_axis(decision_fn, axis=1, arr=y_proba, p=p)
    return f1_score(y_test, y_pred, average="micro")


# %%
scoring_fn(
    y_test,
    y_pred,
    p=np.array([0.0, 0.1, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05]),
)
# %%
scoring_fn(
    y_test,
    y_pred,
    p=np.array([100] * N_CLASSES),
)

# %%
scoring_fn(
    y_test,
    y_pred,
    p=np.array([1 / N_CLASSES] * N_CLASSES),
)

# %%
x0 = np.array([1 / N_CLASSES] * N_CLASSES)
# x0 = [0.0, 0.1, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05]

scoring_lambda = lambda x: scoring_fn(y_test, y_pred, x)
res = minimize(
    scoring_lambda, x0, method='COBYLA'
    )
# %%
res.x
# %%
