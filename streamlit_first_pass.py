import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly.express as px
from sklearn.utils.class_weight import compute_sample_weight

# load data
df = pd.read_csv("mlb_dataset.csv")

# fastballs for testing
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

# xgb params
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(labler.classes_),
    eval_metric= 'mlogloss',     
    random_state=42
)

# weight the samples so class dominates the rest
sample_weights = compute_sample_weight(class_weight='balanced', y=y_labeled)

# fit the model
model.fit(X, y_labeled)

# Returns shape (n_samples, 5) — one prob per class
def get_probs(feature_1, feature_2):
    inputs = pd.DataFrame([[feature_1, feature_2]], columns=features)
    probs = model.predict_proba(inputs)[0]
    return dict(zip(labler.classes_, probs))

# This is currently hard coded but will need to eventually be dynamic/dropdown
plate_x = st.slider("plate_x (left/right)", -2.0, 2.0, 0.0, step=0.05)
plate_z = st.slider("plate_z (height)", 0.5, 5.0, 2.5, step=0.05)

probs = get_probs(plate_x, plate_z)

fig = px.bar(
    x=list(probs.keys()),
    y=list(probs.values()),
    labels={'x': 'outcome', 'y': 'probability'},
    range_y=[0, 1]
)
st.plotly_chart(fig, use_container_width=True)