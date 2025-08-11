import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
df['Target'] = y

# Try different depths
best_score = 0
best_depth = None

for depth in range(2, 21):  # try depths from 2 to 20
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )
    scores = cross_val_score(rf, df[['Feature1', 'Feature2', 'Feature3', 'Feature4']], df['Target'], cv=5)
    mean_score = np.mean(scores)

    print(f"Depth: {depth}, Mean Accuracy: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_depth = depth

print("\nBest Depth:", best_depth)
print("Best Accuracy:", best_score)


# Depth: 2, Mean Accuracy: 0.9030
# Depth: 3, Mean Accuracy: 0.9260
# Depth: 4, Mean Accuracy: 0.9370
# Depth: 5, Mean Accuracy: 0.9490
# Depth: 6, Mean Accuracy: 0.9530
# Depth: 7, Mean Accuracy: 0.9510
# Depth: 8, Mean Accuracy: 0.9510
# Depth: 9, Mean Accuracy: 0.9510
# Depth: 10, Mean Accuracy: 0.9520
# Depth: 11, Mean Accuracy: 0.9530
# Depth: 12, Mean Accuracy: 0.9510
# Depth: 13, Mean Accuracy: 0.9520
# Depth: 14, Mean Accuracy: 0.9500
# Depth: 15, Mean Accuracy: 0.9500
# Depth: 16, Mean Accuracy: 0.9500
# Depth: 17, Mean Accuracy: 0.9500
# Depth: 18, Mean Accuracy: 0.9500
# Depth: 19, Mean Accuracy: 0.9500
# Depth: 20, Mean Accuracy: 0.9500

# Best Depth: 6
# Best Accuracy: 0.953
