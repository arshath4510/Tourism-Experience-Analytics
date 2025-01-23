import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Function to generate sample hotel data
def generate_hotel_data(n_hotels=100):
    np.random.seed(42)
    cities = ['New York', 'Paris', 'Tokyo', 'London', 'Sydney']
    hotel_types = ['Luxury', 'Budget', 'Mid-range', 'Resort', 'Boutique']
    
    data = {
        'name': [f'Hotel {i}' for i in range(1, n_hotels + 1)],
        'city': np.random.choice(cities, n_hotels),
        'type': np.random.choice(hotel_types, n_hotels),
        'price': np.random.randint(50, 500, n_hotels),
        'rating': np.random.uniform(1, 5, n_hotels),
        'distance_to_center': np.random.uniform(0, 10, n_hotels),
        'n_attractions_nearby': np.random.randint(1, 20, n_hotels)
    }
    
    return pd.DataFrame(data)

# Function to get hotel recommendations
def get_recommendations(df, city, hotel_type, budget, n_recommendations=5):
    # Filter by city and hotel type
    filtered_df = df[(df['city'] == city) & (df['type'] == hotel_type) & (df['price'] <= budget)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['price', 'rating', 'distance_to_center', 'n_attractions_nearby']
    normalized_features = scaler.fit_transform(filtered_df[numerical_features])
    
    # Calculate similarity
    similarity = cosine_similarity(normalized_features)
    
    # Get top recommendations
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
    
    return filtered_df.iloc[top_indices]

# Streamlit app
st.title("Hotel Recommendation System")

# Generate sample data
hotels_df = generate_hotel_data()

# User inputs
city = st.selectbox("Select your destination:", hotels_df['city'].unique())
hotel_type = st.selectbox("Select hotel type:", hotels_df['type'].unique())
budget = st.slider("Select your budget (per night):", 
                   min_value=int(hotels_df['price'].min()), 
                   max_value=int(hotels_df['price'].max()), 
                   value=int(hotels_df['price'].median()))

# Get recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(hotels_df, city, hotel_type, budget)
    
    if recommendations.empty:
        st.write("No hotels found matching your criteria. Please try adjusting your preferences.")
    else:
        st.subheader("Recommended Hotels:")
        for _, hotel in recommendations.iterrows():
            st.write(f"**{hotel['name']}**")
            st.write(f"Type: {hotel['type']}")
            st.write(f"Price: ${hotel['price']} per night")
            st.write(f"Rating: {hotel['rating']:.1f}/5.0")
            st.write(f"Distance to city center: {hotel['distance_to_center']:.1f} km")
            st.write(f"Number of attractions nearby: {hotel['n_attractions_nearby']}")
            st.write("---")

# Display sample data
st.sidebar.subheader("Sample Hotel Data")
st.sidebar.dataframe(hotels_df)