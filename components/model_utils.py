import pandas as pd
import streamlit as st
import joblib


# Cache and load pitching model
@st.cache_resource
def load_pitching_model():
    model_a = joblib.load('models/pitch_outcome_model.joblib')
    labeler_a = joblib.load('models/pitch_outcome_labeler.joblib')
    return model_a, labeler_a

# Cache and load batting model
@st.cache_resource
def load_batting_model():
    model_b = joblib.load('models/batted_outcome_model.joblib')
    labeler_b = joblib.load('models/batted_outcome_labler.joblib')
    return model_b, labeler_b

def predict_pitch_outcome(model, labeler, user_inputs):
    inputs = dict(user_inputs)
    inputs['stand'] = 1 if inputs['stand'] == 'R' else 0
    inputs['p_throws'] = 1 if inputs['p_throws'] == 'R' else 0
    
    df = pd.DataFrame([inputs])
    df['pitch_name'] = df['pitch_name'].astype('category')
    
    probs = model.predict_proba(df)[0]
    
    return dict(zip(labeler.classes_, probs))


def predict_batted_outcome(model, labeler, user_inputs, batted_inputs):
    combined = dict(user_inputs)
    combined['stand'] = 1 if combined['stand'] == 'R' else 0
    combined['p_throws'] = 1 if combined['p_throws'] == 'R' else 0
    combined.update(batted_inputs)

    # Column order must match training
    column_order = [
        'pitch_name', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
        'pfx_x', 'pfx_z', 'balls', 'strikes', 'stand', 'p_throws',
        'release_extension', 'arm_angle', 'launch_speed_angle', 'bb_type',
        'swing_length', 'attack_angle', 'bat_speed', 'swing_path_tilt'
    ]

    df = pd.DataFrame([combined])[column_order]
    df['pitch_name'] = df['pitch_name'].astype('category')
    df['bb_type'] = df['bb_type'].astype('category')

    probs = model.predict_proba(df)[0]

    return dict(zip(labeler.classes_, probs))
