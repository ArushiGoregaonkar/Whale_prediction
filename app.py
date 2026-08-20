import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Payer Conversion Predictor", layout="wide")

# --- Title and Description ---
st.title("🎮 Free-to-Play Payer Conversion Predictor")
st.markdown("""
This app predicts the probability that a new free-to-play mobile game player will convert into a paying customer ('whale').
Enter the player's early activity profile below to get a prediction.
""")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("final_payer_conversion_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'final_payer_conversion_model.pkl' not found. Please ensure the model has been trained and saved.")
        st.stop()

model = load_model()

# --- Sidebar with Model Info ---
st.sidebar.header("About the Model")
st.sidebar.markdown("""
- **Problem Type:** Binary Classification (Payer / Non-Payer)
- **Algorithm:** The best performing model from a set of tuned classifiers (e.g., Random Forest, XGBoost)
- **Training Data:** Player activity data
- **Target:** `converted_to_payer`
- **Metric:** ROC-AUC (Area Under the Curve)
""")

# --- Main User Input Section ---
st.subheader("📊 Player Profile Inputs")
st.markdown("Fill in the player's early-game stats and behaviors.")

# Create two columns for better layout
col1, col2 = st.columns(2)

# The features are based on the notebook's features after dropping 'player_id' and 'converted_to_payer'
# All features are included. Widgets are chosen to match data types.

with col1:
    age = st.number_input("Age", min_value=13, max_value=60, value=25, step=1, help="Player's age.")
    gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'], index=0, help="Player's gender.")
    country = st.selectbox("Country", options=['USA', 'India', 'Brazil', 'UK', 'Germany', 'Japan', 'Canada', 'Mexico', 'Indonesia', 'Philippines'], index=0, help="Player's country.")
    acquisition_channel = st.selectbox("Acquisition Channel", options=['organic', 'paid_social', 'paid_search', 'referral', 'influencer'], index=0, help="How the player was acquired.")
    device_type = st.selectbox("Device Type", options=['Android', 'iOS'], index=0, help="Mobile device platform.")
    days_since_install = st.number_input("Days Since Install", min_value=1, max_value=90, value=10, step=1, help="How many days has it been since the first install?")

with col2:
    sessions_last_7d = st.number_input("Sessions (Last 7 Days)", min_value=0, max_value=26, value=7, step=1, help="Number of sessions in the last 7 days.")
    avg_session_length_min = st.number_input("Avg. Session Length (min)", min_value=0.0, max_value=33.0, value=10.0, step=0.5, help="Average session length in minutes.")
    total_playtime_hours = st.number_input("Total Playtime (Hours)", min_value=0.0, max_value=112.0, value=2.0, step=0.5, help="Total hours played.")
    levels_completed = st.number_input("Levels Completed", min_value=0, max_value=51, value=10, step=1, help="Number of levels completed.")
    current_level = st.number_input("Current Level", min_value=1, max_value=51, value=10, step=1, help="Current level of the player.")
    tutorial_completed = st.radio("Tutorial Completed?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No", help="Has the player completed the tutorial?")

# More features in a new section
st.subheader("📈 Engagement & Monetization Signals")
col3, col4 = st.columns(2)

with col3:
    num_friends_connected = st.number_input("Friends Connected", min_value=0, max_value=15, value=2, step=1, help="Number of friends connected.")
    push_notifications_enabled = st.radio("Push Notifications Enabled?", options=[1, 0], index=0, format_func=lambda x: "Yes" if x == 1 else "No", help="Did the player enable push notifications?")
    ad_views = st.number_input("Ad Views", min_value=0, max_value=24, value=6, step=1, help="Number of video ads viewed.")
    rewarded_ad_views = st.number_input("Rewarded Ad Views", min_value=0, max_value=14, value=2, step=1, help="Number of rewarded video ads viewed.")

with col4:
    store_visits = st.number_input("Store Visits", min_value=0, max_value=12, value=2, step=1, help="Number of store visits.")
    items_viewed_in_store = st.number_input("Items Viewed in Store", min_value=0, max_value=41, value=5, step=1, help="Number of items viewed in the store.")
    wishlist_items = st.number_input("Wishlist Items", min_value=0, max_value=9, value=1, step=1, help="Number of items in the wishlist.")
    days_active_last_30 = st.number_input("Days Active (Last 30)", min_value=0, max_value=29, value=15, step=1, help="Number of active days in the last 30 days.")

with col3:
    streak_days = st.number_input("Current Streak (Days)", min_value=0, max_value=46, value=2, step=1, help="Current daily login streak.")
    rage_quit_events = st.number_input("Rage Quit Events", min_value=0, max_value=14, value=3, step=1, help="Number of 'rage quit' events.")
    level_fail_rate = st.slider("Level Fail Rate", min_value=0.0, max_value=1.0, value=0.6, step=0.01, help="Failure rate at levels.")
    social_shares = st.number_input("Social Shares", min_value=0, max_value=7, value=1, step=1, help="Number of social shares.")

# --- Prepare Data for Prediction ---
# The order of features must exactly match the training data.
# Based on the notebook, the order after dropping 'player_id' is:
input_data = {
    'age': age,
    'gender': gender,
    'country': country,
    'acquisition_channel': acquisition_channel,
    'device_type': device_type,
    'days_since_install': days_since_install,
    'sessions_last_7d': sessions_last_7d,
    'avg_session_length_min': avg_session_length_min,
    'total_playtime_hours': total_playtime_hours,
    'levels_completed': levels_completed,
    'current_level': current_level,
    'tutorial_completed': tutorial_completed,
    'num_friends_connected': num_friends_connected,
    'push_notifications_enabled': push_notifications_enabled,
    'ad_views': ad_views,
    'rewarded_ad_views': rewarded_ad_views,
    'store_visits': store_visits,
    'items_viewed_in_store': items_viewed_in_store,
    'wishlist_items': wishlist_items,
    'days_active_last_30': days_active_last_30,
    'streak_days': streak_days,
    'rage_quit_events': rage_quit_events,
    'level_fail_rate': level_fail_rate,
    'social_shares': social_shares
}

input_df = pd.DataFrame([input_data])

# --- Prediction Button ---
if st.button("Predict Conversion Probability"):
    # Use the loaded model to make a prediction
    try:
        # The model is a pipeline, so it will handle preprocessing automatically
        prediction_proba = model.predict_proba(input_df)

        # Extract probabilities for class 1 (payer)
        prob_payer = prediction_proba[0][1]

        # --- Display Results ---
        st.subheader("📈 Prediction Result")
        st.markdown(f"### The player has a **{prob_payer:.1%}** probability of converting to a paying customer.")

        # Add a visual gauge or progress bar
        st.progress(prob_payer)

        if prob_payer > 0.7:
            st.success("✅ High probability. Consider targeting this player with special offers.")
        elif prob_payer > 0.4:
            st.warning("⚠️ Moderate probability. Further engagement may be needed to encourage conversion.")
        else:
            st.info("ℹ️ Low probability. Monitor engagement but don't over-invest in this player yet.")

        # Optional: Show raw probabilities
        with st.expander("🔎 Raw Probabilities"):
            st.write(f"Probability of Non-Payer: **{prediction_proba[0][0]:.4f}**")
            st.write(f"Probability of Payer: **{prediction_proba[0][1]:.4f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check that all inputs are valid and the model file is correct.")
