import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="NYC Airbnb Room Type Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished modern look, card containers, and typography
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF5A5F;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #E0484D;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        text-align: center;
        border-top: 4px solid #FF5A5F;
    }
    h1, h2, h3 {
        color: #222222;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cached Data Loading & Model Training
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("AB_NYC_2019.csv")
    except FileNotFoundError:
        return None, None, None, None

    features = [
        'neighbourhood_group', 'latitude', 'longitude', 
        'price', 'minimum_nights', 'number_of_reviews', 
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
    ]
    
    data = df[features + ['room_type']].dropna().copy()
    data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

    le_neigh = LabelEncoder()
    data['neighbourhood_group_encoded'] = le_neigh.fit_transform(data['neighbourhood_group'])
    
    le_room = LabelEncoder()
    data['room_type_encoded'] = le_room.fit_transform(data['room_type'])

    X = data[['neighbourhood_group_encoded', 'latitude', 'longitude', 'price', 
              'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
              'calculated_host_listings_count', 'availability_365']]
    y = data['room_type_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    return model, le_neigh, le_room, df

model, le_neigh, le_room, raw_df = load_and_train_model()

if model is None:
    st.error("🚨 Error: 'AB_NYC_2019.csv' not found in the root directory. Please make sure the dataset file is present.")
else:
    # App Header Banner
    st.markdown("""
    <div style='padding: 20px 0; border-bottom: 1px solid #eaeaea; margin-bottom: 25px;'>
        <h1 style='margin:0; font-size: 2.2rem;'>🏠 NYC Airbnb Room Type Intelligence Hub</h1>
        <p style='color: #666; font-size: 1.1rem; margin-top: 5px;'>Predict listing categories instantly with machine learning and explore NYC open housing dynamics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Tabs
    tab_pred, tab_insights = st.tabs(["🔮 Room Type Predictor", "📊 Dataset Explorer"])

    with tab_pred:
        # Sidebar Controls
        st.sidebar.markdown("### 🎛️ Listing Configuration")
        st.sidebar.markdown("Adjust parameters to simulate a listing:")

        selected_neigh = st.sidebar.selectbox("Borough / Neighbourhood Group", list(le_neigh.classes_))
        neigh_encoded = le_neigh.transform([selected_neigh])[0]

        price = st.sidebar.slider("Price per Night ($)", min_value=10, max_value=1500, value=120, step=5)
        minimum_nights = st.sidebar.slider("Minimum Nights", min_value=1, max_value=90, value=2, step=1)
        number_of_reviews = st.sidebar.number_input("Total Reviews", min_value=0, max_value=400, value=15, step=1)
        reviews_per_month = st.sidebar.slider("Reviews per Month", min_value=0.0, max_value=15.0, value=0.8, step=0.1)
        host_listings = st.sidebar.number_input("Host Total Listings", min_value=1, max_value=50, value=1, step=1)
        availability = st.sidebar.slider("Availability (Days/Year)", min_value=0, max_value=365, value=200)

        # Approximate coordinate centers for NYC Boroughs
        borough_coords = {
            'Manhattan': (40.7831, -73.9712),
            'Brooklyn': (40.6782, -73.9442),
            'Queens': (40.7282, -73.7949),
            'Bronx': (40.8448, -73.8648),
            'Staten Island': (40.5795, -74.1502)
        }
        lat, lon = borough_coords.get(selected_neigh, (40.7128, -74.0060))

        # Main Prediction Column Layout
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("### 📋 Input Summary")
            input_summary_df = pd.DataFrame({
                "Feature": ["Borough", "Nightly Price", "Min Nights", "Reviews", "Reviews/Month", "Host Listings", "Availability"],
                "Value": [selected_neigh, f"${price}", minimum_nights, number_of_reviews, reviews_per_month, host_listings, f"{availability} days"]
            })
            st.dataframe(input_summary_df, use_container_width=True, hide_index=True)

            predict_btn = st.button("🚀 Run Room Type Prediction")

        with col_right:
            st.markdown("### 🎯 Model Output")
            if predict_btn:
                input_data = np.array([[
                    neigh_encoded, lat, lon, price, minimum_nights, 
                    number_of_reviews, reviews_per_month, host_listings, availability
                ]])

                prediction_encoded = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                predicted_room_type = le_room.inverse_transform([prediction_encoded])[0]

                # Metric Display Card
                st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='color: #666; margin-bottom: 0;'>Predicted Category</h4>
                    <h2 style='color: #FF5A5F; margin-top: 5px; font-size: 2rem;'>{predicted_room_type}</h2>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📈 Probability Confidence Breakdown")
                for r_type, prob in zip(le_room.classes_, probabilities):
                    st.markdown(f"**{r_type}** ({prob*100:.1f}%)")
                    st.progress(float(prob))
            else:
                st.info("👈 Configure parameters on the left sidebar and click **'Run Room Type Prediction'** to generate results.")

    with tab_insights:
        st.markdown("### 📊 Dataset Quick Overview & Analytics")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Listings Analyzed", f"{len(raw_df):,}")
        with m2:
            st.metric("Average Price / Night", f"${raw_df['price'].mean():.2f}")
        with m3:
            st.metric("Boroughs Covered", f"{raw_df['neighbourhood_group'].nunique()}")

        st.markdown("#### 🗺️ Listing Distribution Map (Sample)")
        map_sample = raw_df[['latitude', 'longitude', 'price']].dropna().sample(min(1000, len(raw_df)))
        st.map(map_sample, latitude='latitude', longitude='longitude', size=2, color='#FF5A5F')