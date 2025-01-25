import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def load_data(self, file_path):
        """Load and preprocess the news dataset."""
        df = pd.read_csv(file_path)
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training."""
        # Combine title and text for feature extraction
        df['combined_text'] = df['News_Title'] + ' ' + df['News_Text']
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X = df['combined_text']
        y = self.label_encoder.fit_transform(df['Label'])
        return train_test_split(X, y, test_size=test_size, random_state=random_state) 