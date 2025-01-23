import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    # Load data
    user_data = pd.read_csv('user_data.csv')
    attraction_data = pd.read_csv('attraction_data.csv')
    transaction_data = pd.read_csv('transaction_data.csv')
    
    # Merge datasets
    merged_data = pd.merge(transaction_data, user_data, on='UserId')
    merged_data = pd.merge(merged_data, attraction_data, on='AttractionId')
    
    # Feature engineering
    merged_data['VisitSeason'] = pd.to_datetime(merged_data['VisitYear'].astype(str) + '-' + merged_data['VisitMonth'].astype(str) + '-01').dt.month.map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    return merged_data

def train_classification_model(data):
    # Prepare features and target
    features = ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 'VisitSeason', 'ContinentId', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId', 'Rating']
    X = data[features]
    y = data['VisitMode']
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing steps
    numeric_features = ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 'Rating']
    categorical_features = ['VisitSeason', 'ContinentId', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create a pipeline with preprocessor and model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', XGBClassifier(random_state=42))])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate feature importance
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    return model, X_test, y_test, y_pred, cv_scores, feature_importance, feature_names, le

def plot_feature_importance(feature_importance, feature_names):
    # Sort features by importance
    indices = np.argsort(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(len(feature_importance)), feature_importance[indices])
    plt.yticks(range(len(feature_importance)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    return plt

if __name__ == "__main__":
    data = load_and_preprocess_data()
    model, X_test, y_test, y_pred, cv_scores, feature_importance, feature_names, le = train_classification_model(data)
    
    # Print cross-validation results
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot feature importance
    plt = plot_feature_importance(feature_importance, feature_names)
    plt.show()
