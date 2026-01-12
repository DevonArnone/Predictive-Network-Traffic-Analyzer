"""
Feature Engineering and Preprocessing Module
Handles windowing, normalization, and data preparation for LSTM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import json

class TrafficFeatureEngineer:
    def __init__(self, window_size_seconds=5):
        """
        Initialize feature engineer
        
        Args:
            window_size_seconds: Size of time window for aggregation (default: 5 seconds)
        """
        self.window_size_seconds = window_size_seconds
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def json_to_dataframe(self, json_data: List[dict]) -> pd.DataFrame:
        """
        Convert list of JSON packet records to DataFrame
        
        Args:
            json_data: List of packet dictionaries
            
        Returns:
            DataFrame with parsed network traffic data
        """
        df = pd.DataFrame(json_data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def create_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group packets into time windows and calculate aggregate features
        
        Args:
            df: DataFrame with packet data
            
        Returns:
            DataFrame with windowed features
        """
        # Set timestamp as index for time-based grouping
        df_indexed = df.set_index('timestamp')
        
        # Resample into windows and aggregate
        windowed = df_indexed.resample(f'{self.window_size_seconds}s').agg({
            'packet_length': ['sum', 'mean', 'count'],
            'inter_arrival_time': 'mean',
            'dest_port': lambda x: x.mode()[0] if len(x.mode()) > 0 else 0,  # Most common port
            'protocol': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'TCP'  # Most common protocol
        })
        
        # Flatten column names
        windowed.columns = ['total_bytes', 'avg_packet_size', 'packet_count', 
                           'avg_inter_arrival_time', 'common_dest_port', 'common_protocol']
        
        # Reset index to have timestamp as column
        windowed = windowed.reset_index()
        
        # Drop rows with no packets (empty windows)
        windowed = windowed[windowed['packet_count'] > 0]
        
        # Encode protocol as numeric (simple encoding)
        protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
        windowed['protocol_encoded'] = windowed['common_protocol'].map(protocol_map)
        
        return windowed
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare features for model training
        
        Args:
            df: Windowed DataFrame
            
        Returns:
            DataFrame with selected features
        """
        # Select features for model
        features = df[[
            'total_bytes',
            'avg_packet_size',
            'packet_count',
            'avg_inter_arrival_time',
            'protocol_encoded'
        ]].copy()
        
        return features
    
    def normalize_features(self, features: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Normalize features using MinMaxScaler
        
        Args:
            features: DataFrame with features to normalize
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Normalized numpy array
        """
        if fit:
            normalized = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming inference data")
            normalized = self.scaler.transform(features)
        
        return normalized
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training (sliding window)
        
        Args:
            data: Normalized feature array
            sequence_length: Number of timesteps to look back
            
        Returns:
            Tuple of (X, y) where:
            - X: Sequences of shape (samples, sequence_length, features)
            - y: Targets of shape (samples, 1) - predicting packet_count
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            # Input sequence: last 'sequence_length' windows
            X.append(data[i:i + sequence_length])
            # Target: packet_count of next window (index 2 is packet_count in our feature set)
            y.append(data[i + sequence_length, 2])  # packet_count is at index 2
        
        return np.array(X), np.array(y)
    
    def process_for_training(self, json_data: List[dict], sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline for training data preparation
        
        Args:
            json_data: List of packet dictionaries
            sequence_length: Number of timesteps for LSTM
            
        Returns:
            Tuple of (X, y) ready for model training
        """
        # Convert to DataFrame
        df = self.json_to_dataframe(json_data)
        
        # Create windows
        windowed = self.create_windows(df)
        
        # Select features
        features = self.select_features(windowed)
        
        # Normalize (fit scaler)
        normalized = self.normalize_features(features, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(normalized, sequence_length)
        
        return X, y
    
    def process_for_inference(self, json_data: List[dict], sequence_length: int = 10) -> np.ndarray:
        """
        Complete pipeline for inference data preparation
        
        Args:
            json_data: List of packet dictionaries
            sequence_length: Number of timesteps for LSTM
            
        Returns:
            X array ready for model inference (shape: (1, sequence_length, features))
        """
        # Convert to DataFrame
        df = self.json_to_dataframe(json_data)
        
        # Create windows
        windowed = self.create_windows(df)
        
        # Select features
        features = self.select_features(windowed)
        
        # Normalize (transform only, no fitting)
        normalized = self.normalize_features(features, fit=False)
        
        # Take last sequence_length windows
        if len(normalized) < sequence_length:
            # Pad with zeros if not enough data
            padding = np.zeros((sequence_length - len(normalized), normalized.shape[1]))
            normalized = np.vstack([padding, normalized])
        
        # Get last sequence
        X = normalized[-sequence_length:].reshape(1, sequence_length, -1)
        
        return X

