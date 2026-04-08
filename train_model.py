import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")

# ── 2. Encode categorical columns ─────────────────────────────────────────────
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].replace(df[col].unique(), range(df[col].nunique()))

# ── 3. Fix zero values in Cholesterol and RestingBP using KNN imputation ──────
for col in ['Cholesterol', 'RestingBP']:
    df[col] = df[col].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=3)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Cast all columns except Oldpeak to int
for col in df.columns.drop('Oldpeak'):
    df[col] = df[col].astype(int)

# ── 4. Split features and target ──────────────────────────────────────────────
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Save the scaler so the app can use it
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# ── 5. Define models ──────────────────────────────────────────────────────────
models = {
    'LogisticRegression':       LogisticRegression(C=1, solver='lbfgs', max_iter=500),
    'svm':                      SVC(C=10, gamma='scale', kernel='rbf'),
    'DecisionTreeClassifier':   DecisionTreeClassifier(max_depth=5, min_samples_split=3,
                                                       min_samples_leaf=2, random_state=42,
                                                       class_weight='balanced'),
    'RandomForestClassifier':   RandomForestClassifier(n_estimators=300, max_depth=10,
                                                       random_state=42),
    'XGBClassifier':            XGBClassifier(n_estimators=300, max_depth=5,
                                              learning_rate=0.05, random_state=42,
                                              eval_metric='logloss'),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=300,
                                                             learning_rate=0.05,
                                                             max_depth=4, random_state=42),
}

# ── 6. Train, evaluate, and save each model ───────────────────────────────────
print(f"{'Model':<30} {'Accuracy':>10}")
print("-" * 42)

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<30} {acc*100:>9.2f}%")
    pickle.dump(model, open(f'{name}.pkl', 'wb'))

print("\nAll models and scaler saved successfully.")
