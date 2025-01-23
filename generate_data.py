import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

def generate_user_data(num_users=1000):
    continents = ['North America', 'South America', 'Europe', 'Asia', 'Africa', 'Australia']
    regions = ['North', 'South', 'East', 'West', 'Central']
    countries = ['USA', 'Canada', 'UK', 'France', 'Germany', 'Japan', 'Australia', 'Brazil', 'India', 'South Africa']
    cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Rio de Janeiro', 'Mumbai', 'Cape Town', 'Berlin', 'Toronto']

    data = {
        'UserId': range(1, num_users + 1),
        'ContinentId': [random.randint(1, len(continents)) for _ in range(num_users)],
        'RegionId': [random.randint(1, len(regions)) for _ in range(num_users)],
        'CountryId': [random.randint(1, len(countries)) for _ in range(num_users)],
        'CityId': [random.randint(1, len(cities)) for _ in range(num_users)]
    }
    return pd.DataFrame(data)

def generate_attraction_data(num_attractions=500):
    attraction_types = ['Beach', 'Museum', 'Park', 'Historical Site', 'Restaurant', 'Theme Park']
    cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Rio de Janeiro', 'Mumbai', 'Cape Town', 'Berlin', 'Toronto']

    data = {
        'AttractionId': range(1, num_attractions + 1),
        'AttractionCityId': [random.randint(1, len(cities)) for _ in range(num_attractions)],
        'AttractionTypeId': [random.randint(1, len(attraction_types)) for _ in range(num_attractions)],
        'Attraction': [fake.company() for _ in range(num_attractions)],
        'AttractionAddress': [fake.address() for _ in range(num_attractions)]
    }
    return pd.DataFrame(data)

def generate_transaction_data(num_transactions=10000, num_users=1000, num_attractions=500):
    visit_modes = ['Business', 'Couples', 'Family', 'Friends', 'Solo']

    data = {
        'TransactionId': range(1, num_transactions + 1),
        'UserId': [random.randint(1, num_users) for _ in range(num_transactions)],
        'VisitYear': [random.randint(2018, 2023) for _ in range(num_transactions)],
        'VisitMonth': [random.randint(1, 12) for _ in range(num_transactions)],
        'VisitMode': [random.choice(visit_modes) for _ in range(num_transactions)],
        'AttractionId': [random.randint(1, num_attractions) for _ in range(num_transactions)],
        'Rating': [round(random.uniform(1, 5), 1) for _ in range(num_transactions)]
    }
    return pd.DataFrame(data)

def generate_all_data():
    user_data = generate_user_data()
    attraction_data = generate_attraction_data()
    transaction_data = generate_transaction_data()

    user_data.to_csv('user_data.csv', index=False)
    attraction_data.to_csv('attraction_data.csv', index=False)
    transaction_data.to_csv('transaction_data.csv', index=False)

    print("Data generation complete. Files saved: user_data.csv, attraction_data.csv, transaction_data.csv")

if __name__ == "__main__":
    generate_all_data()

