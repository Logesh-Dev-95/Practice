import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Dataset
from sklearn.datasets import make_classification

# Generate a bigger dataset
X, y = make_classification(
    n_samples=100000,   # number of rows
    n_features=4,    # number of features
    n_informative=3, # how many features are actually useful
    n_redundant=1,   # redundant features (linear combos)
    n_classes=2,     # binary classification
    random_state=42
)

# Convert to DataFrame for readability
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
df['Target'] = y

df.head()

X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4']]
y = df['Target']


# Test depths from 1 to 10
best_depth = None
best_score = 0

print("Depth | Mean Accuracy (5-fold CV)")
print("---------------------------------")

for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    mean_score = np.mean(scores)
    print(f"{depth:5d} | {mean_score:.3f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_depth = depth

print("\nBest depth:", best_depth, "with accuracy:", round(best_score, 3))


# Depth | Mean Accuracy (5-fold CV)
# ---------------------------------
#     1 | 0.893
#     2 | 0.895
#     3 | 0.908
#     4 | 0.915
#     5 | 0.931
#     6 | 0.931
#     7 | 0.934
#     8 | 0.935
#     9 | 0.935
#    10 | 0.936
#    11 | 0.932
#    12 | 0.932
#    13 | 0.932
#    14 | 0.932
#    15 | 0.932
#    16 | 0.932
#    17 | 0.932
#    18 | 0.932
#    19 | 0.932
#    20 | 0.932

# Best depth: 10 with accuracy: 0.936
