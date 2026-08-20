"""
Mobile Game Whale Prediction - Streamlit App

This app predicts whether a free-to-play mobile game player is likely to 
convert to a paying customer based on early player behavior data.

Author: ML Model Deployment
Date: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Whale Prediction App",
    page_icon="🐋",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .prediction-negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .probability-bar {
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feature-importance {
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and metadata."""
    model_path = "models/whale_prediction_model.pkl"
    metadata_path = "models/model_metadata.json"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.info("Please ensure the model has been trained and saved correctly.")
        return None, None
    
    try:
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def create_input_widgets(metadata):
    """Create input widgets based on feature types and metadata."""
    inputs = {}
    
    # Determine which features are numerical and categorical
    numerical_features = metadata.get('numerical_features', [])
    categorical_features = metadata.get('categorical_features', [])
    
    st.markdown("### Player Information")
    
    col1, col2 = st.columns(2)
    
    # Numerical features (with appropriate ranges based on dataset)
    num_widgets = [
        ('age', 13, 60, 27, "Age"),
        ('days_since_install', 1, 90, 15, "Days Since Install"),
        ('sessions_last_7d', 0, 26, 7, "Sessions in Last 7 Days"),
        ('avg_session_length_min', 0.5, 35.0, 10.5, "Avg Session Length (minutes)"),
        ('total_playtime_hours', 0.0, 115.0, 3.0, "Total Playtime (hours)"),
        ('levels_completed', 0, 55, 14, "Levels Completed"),
        ('current_level', 1, 55, 14, "Current Level"),
        ('num_friends_connected', 0, 15, 2, "Friends Connected"),
        ('ad_views', 0, 25, 6, "Ad Views"),
        ('rewarded_ad_views', 0, 15, 2, "Rewarded Ad Views"),
        ('store_visits', 0, 12, 2, "Store Visits"),
        ('items_viewed_in_store', 0, 42, 5, "Items Viewed in Store"),
        ('wishlist_items', 0, 10, 1, "Wishlist Items"),
        ('days_active_last_30', 0, 29, 15, "Days Active (Last 30)"),
        ('streak_days', 0, 46, 4, "Streak Days"),
        ('rage_quit_events', 0, 14, 3, "Rage Quit Events"),
        ('level_fail_rate', 0.01, 1.0, 0.60, "Level Fail Rate"),
        ('social_shares', 0, 7, 1, "Social Shares"),
    ]
    
    # Create numerical inputs
    for i, (col, min_val, max_val, default, label) in enumerate(num_widgets):
        with col1 if i % 2 == 0 else col2:
            if col in numerical_features or col not in categorical_features:
                inputs[col] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default),
                    step=0.1 if '.' in str(min_val) else 1.0,
                    help=f"Range: {min_val} - {max_val}"
                )
    
    # Categorical features
    st.markdown("### Player Demographics & Behavior")
    
    cat_cols = st.columns(2)
    
    # Gender
    with cat_cols[0]:
        inputs['gender'] = st.selectbox(
            "Gender",
            options=['Male', 'Female', 'Other'],
            help="Player's gender"
        )
    
    # Country - common countries from dataset
    countries = ['USA', 'India', 'Brazil', 'Indonesia', 'Philippines', 'Mexico', 
                 'UK', 'Germany', 'Canada', 'Japan']
    with cat_cols[1]:
        inputs['country'] = st.selectbox(
            "Country",
            options=countries,
            help="Player's country"
        )
    
    # Acquisition Channel
    channels = ['organic', 'paid_social', 'paid_search', 'referral', 'influencer']
    with cat_cols[0]:
        inputs['acquisition_channel'] = st.selectbox(
            "Acquisition Channel",
            options=channels,
            help="How the player acquired the game"
        )
    
    # Device Type
    with cat_cols[1]:
        inputs['device_type'] = st.selectbox(
            "Device Type",
            options=['Android', 'iOS'],
            help="Player's device type"
        )
    
    # Boolean features
    st.markdown("### Settings & Engagement")
    bool_cols = st.columns(2)
    
    with bool_cols[0]:
        inputs['tutorial_completed'] = st.radio(
            "Tutorial Completed",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0,
            horizontal=True,
            help="Whether the player completed the tutorial"
        )
    
    with bool_cols[1]:
        inputs['push_notifications_enabled'] = st.radio(
            "Push Notifications Enabled",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0,
            horizontal=True,
            help="Whether the player has push notifications enabled"
        )
    
    return inputs


def predict_player(model, input_df):
    """Make prediction on input data."""
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None


def display_prediction(prediction, probability):
    """Display prediction results."""
    st.markdown("---")
    
    if prediction == 1:
        st.markdown(
            f"""
            <div class="prediction-box prediction-positive">
                <h2 style="color: #28a745;">🐋 Likely to Convert (Whale)</h2>
                <p style="font-size: 1.1rem;">The player is predicted to become a paying customer.</p>
                <p style="font-size: 1.2rem; font-weight: bold;">
                    Probability: {probability[1] * 100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Probability bar for positive class
        st.markdown(f"""
            <div style="margin-top: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Non-Payer</span>
                    <span>Payer</span>
                </div>
                <div style="background-color: #e9ecef; border-radius: 10px; overflow: hidden; height: 25px;">
                    <div style="background: linear-gradient(to right, #dc3545, #ffc107, #28a745); height: 100%; width: {probability[1] * 100:.1f}%;"></div>
                </div>
                <div style="display: flex; justify-content: center;">
                    <span style="font-weight: bold;">{probability[1] * 100:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(
            f"""
            <div class="prediction-box prediction-negative">
                <h2 style="color: #dc3545;">🎮 Likely to Stay Free-to-Play</h2>
                <p style="font-size: 1.1rem;">The player is predicted to remain a non-paying user.</p>
                <p style="font-size: 1.2rem; font-weight: bold;">
                    Probability: {probability[0] * 100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display probabilities in a more detailed format
    with st.expander("Detailed Probability Breakdown"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Non-Payer Probability", f"{probability[0] * 100:.1f}%")
        with col2:
            st.metric("Payer Probability", f"{probability[1] * 100:.1f}%")


def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🐋 Mobile Game Whale Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Predict whether a free-to-play player will convert to a paying customer based on early behavior patterns.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Load model and metadata
    model, metadata = load_model()
    
    if model is None or metadata is None:
        st.stop()
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("### ℹ️ Model Information")
        st.markdown(f"**Model Type:** {metadata.get('model_type', 'Unknown')}")
        st.markdown(f"**Best Parameters:**")
        for param, value in metadata.get('best_params', {}).items():
            st.markdown(f"- `{param}`: {value}")
        st.markdown(f"**Training F1 Score:** {metadata.get('cv_score', 'N/A'):.4f}")
        st.markdown(f"**Test F1 Score:** {metadata.get('test_f1', 'N/A'):.4f}")
        st.markdown(f"**Test ROC-AUC:** {metadata.get('test_roc_auc', 'N/A'):.4f}")
        st.markdown(f"**Class Distribution:**")
        class_dist = metadata.get('class_distribution', {})
        for cls, pct in class_dist.items():
            label = "Payer" if cls == 1 else "Non-Payer"
            st.markdown(f"- {label}: {pct*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### 📊 Feature Information")
        st.markdown(f"**Total Features:** {metadata.get('n_features', 'N/A')}")
        st.markdown(f"**Numerical Features:** {len(metadata.get('numerical_features', []))}")
        st.markdown(f"**Categorical Features:** {len(metadata.get('categorical_features', []))}")
    
    # Main content
    with st.container():
        st.markdown("### Enter Player Information")
        st.markdown("Fill in the player's behavior data to predict conversion likelihood.")
        
        # Create input widgets
        input_data = create_input_widgets(metadata)
        
        # Prediction button
        if st.button("🔮 Predict Conversion", type="primary", use_container_width=True):
            # Create DataFrame from inputs
            input_df = pd.DataFrame([input_data])
            
            # Ensure column order matches training
            # Get expected feature order from metadata
            expected_features = metadata.get('feature_names', [])
            
            # If feature names are available, reorder columns
            if expected_features:
                # Map input columns to expected feature names
                # Note: One-hot encoded features will be created by the pipeline
                # We just need to pass the raw features in the correct order
                raw_features = metadata.get('numerical_features', []) + metadata.get('categorical_features', [])
                input_df = input_df[raw_features]
            
            # Make prediction
            prediction, probability = predict_player(model, input_df)
            
            if prediction is not None:
                display_prediction(prediction, probability)
                
                # Additional business insights
                with st.expander("💡 Business Insights"):
                    st.markdown("""
                    **Next Steps If Likely to Convert:**
                    - Consider sending personalized offers or discounts
                    - Engage with targeted push notifications
                    - Provide exclusive content or early access
                    
                    **Next Steps If Not Likely to Convert:**
                    - Focus on improving engagement metrics
                    - Offer value-driven incentives to encourage spending
                    - Monitor behavior for changes in patterns
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Built with Streamlit • Model trained on mobile game player data
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()