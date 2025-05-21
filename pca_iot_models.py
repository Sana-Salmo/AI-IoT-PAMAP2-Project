import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simulated PAMAP2-like dataset: 1000 samples, 54 features
np.random.seed(42)
X_simulated = np.random.randn(1000, 54)

# Simulate missing values (-999.0 replaced with NaN)
mask = np.random.rand(*X_simulated.shape) < 0.05
X_simulated[mask] = np.nan

# Simulate labels (5 activity classes)
y_simulated = np.random.choice([1, 2, 3, 4, 5], size=1000)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_simulated)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# PCA to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_simulated, test_size=0.3, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y_simulated, test_size=0.3, random_state=42)

# Define models
models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

# Evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc_original = accuracy_score(y_test, model.predict(X_test))

    model.fit(X_train_pca, y_train)
    acc_pca = accuracy_score(y_test, model.predict(X_test_pca))

    results.append({
        "Model": name,
        "Accuracy (Original)": round(acc_original * 100, 2),
        "Accuracy (PCA)": round(acc_pca * 100, 2)
    })

results_df = pd.DataFrame(results)
print(results_df)
