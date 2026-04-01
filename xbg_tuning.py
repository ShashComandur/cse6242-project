import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


# load data
df = pd.read_csv("mlb_dataset.csv")

# fastballs only for testing
df = df[df['pitch_name'] == '4-Seam Fastball']

# set features here
features = [
    # 'release_speed',
    # 'release_spin_rate',
    # 'pfx_x',
    # 'pfx_z'
    'plate_x',
    'plate_z'
]
# dependent variable
target = 'event_group'

# create dataset
df = df[features + [target]].dropna()

# split df
X = df[features]
y = df[target]

# this allows us to number each class ('event_group')
labler = LabelEncoder()
y_labeled = labler.fit_transform(y)

# classic train test splt
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled)

# xgb params
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(labler.classes_),
    eval_metric= 'mlogloss',     
    random_state=42
)

# weight the samples so class dominates the rest
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# fit the model/make preds
model.fit(X_train, y_train, sample_weight=sample_weights)
predictions = model.predict(X_test)

# Get accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}\n")


print(classification_report(y_test, predictions, target_names=labler.classes_))
