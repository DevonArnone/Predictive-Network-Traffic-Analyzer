"""
Real-Time Inference Pipeline
Consumes from Kafka, processes data, and runs LSTM predictions
"""

import json
import time
import torch
import numpy as np
import pandas as pd
from confluent_kafka import Consumer
from lstm_model import TrafficLSTM
from feature_engineering import TrafficFeatureEngineer
import pickle
import os

class TrafficInferencePipeline:
    def __init__(self, 
                 kafka_bootstrap_servers='localhost:9092',
                 topic='network_traffic',
                 model_path='traffic_lstm_model.pth',
                 scaler_path='scaler.pkl',
                 sequence_length=10,
                 window_size_seconds=5,
                 alert_threshold=100):
        """
        Initialize inference pipeline
        
        Args:
            kafka_bootstrap_servers: Kafka broker address
            topic: Kafka topic to consume from
            model_path: Path to trained model file
            scaler_path: Path to saved scaler
            sequence_length: Number of timesteps for LSTM
            window_size_seconds: Window size for feature engineering
            alert_threshold: Packet count threshold for alerts
        """
        self.topic = topic
        self.sequence_length = sequence_length
        self.window_size_seconds = window_size_seconds
        self.alert_threshold = alert_threshold
        
        # Initialize Kafka consumer
        self.consumer = Consumer({
            'bootstrap.servers': kafka_bootstrap_servers,
            'group.id': 'inference_consumer',
            'auto.offset.reset': 'latest'  # Start from latest messages
        })
        self.consumer.subscribe([topic])
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
        
        self.model = TrafficLSTM(input_size=5, hidden_size=64, num_layers=2, output_size=1)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}. Please train the model first.")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Initialize feature engineer with loaded scaler
        self.engineer = TrafficFeatureEngineer(window_size_seconds=window_size_seconds)
        self.engineer.scaler = scaler
        self.engineer.is_fitted = True
        
        # Buffer for accumulating packets
        self.packet_buffer = []
        self.last_prediction_time = time.time()
        
    def denormalize_packet_count(self, normalized_value, scaler):
        """
        Denormalize predicted packet count back to original scale
        
        Args:
            normalized_value: Normalized prediction
            scaler: Fitted scaler
            
        Returns:
            Denormalized packet count
        """
        # Create a dummy array with the same shape as features
        dummy = np.zeros((1, 5))
        dummy[0, 2] = normalized_value  # packet_count is at index 2
        
        # Inverse transform
        denormalized = scaler.inverse_transform(dummy)
        
        return denormalized[0, 2]
    
    def process_and_predict(self, packets):
        """
        Process packets and generate prediction
        
        Args:
            packets: List of packet dictionaries
            
        Returns:
            Predicted packet count for next window
        """
        try:
            # Process for inference
            X = self.engineer.process_for_inference(packets, self.sequence_length)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(X_tensor)
                predicted_value = prediction.item()
            
            # Denormalize
            predicted_packet_count = self.denormalize_packet_count(
                predicted_value, 
                self.engineer.scaler
            )
            
            return max(0, predicted_packet_count)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def check_alert(self, predicted_packet_count):
        """
        Check if predicted traffic exceeds threshold
        
        Args:
            predicted_packet_count: Predicted packet count
            
        Returns:
            True if alert should be triggered
        """
        return predicted_packet_count > self.alert_threshold
    
    def send_alert(self, predicted_packet_count):
        """
        Send alert when traffic threshold is exceeded
        
        Args:
            predicted_packet_count: Predicted packet count
        """
        print(f"\nALERT: Traffic threshold exceeded")
        print(f"  Predicted: {predicted_packet_count:.2f} packets")
        print(f"  Threshold: {self.alert_threshold} packets")
        print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # In a production system, this could send WebSocket updates,
        # write to a database, or trigger other actions
    
    def run(self, batch_window_seconds=5):
        """
        Run the inference pipeline
        
        Args:
            batch_window_seconds: How often to process batches and make predictions
        """
        print("Inference Pipeline Started")
        print(f"  Topic: {self.topic}")
        print(f"  Window: {self.window_size_seconds}s | Threshold: {self.alert_threshold} packets")
        print("Listening for traffic...\n")
        
        try:
            while True:
                # Collect messages for batch window
                batch_start = time.time()
                batch_packets = []
                
                while time.time() - batch_start < batch_window_seconds:
                    msg = self.consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        continue
                    if msg.error():
                        print(f"Consumer error: {msg.error()}")
                        continue
                    
                    try:
                        packet = json.loads(msg.value().decode('utf-8'))
                        batch_packets.append(packet)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding message: {e}")
                        continue
                
                # Add to buffer
                self.packet_buffer.extend(batch_packets)
                
                # Keep only recent packets (last 2 minutes worth)
                current_time = time.time()
                cutoff_time = current_time - (2 * 60)  # 2 minutes
                
                self.packet_buffer = [
                    p for p in self.packet_buffer
                    if pd.to_datetime(p['timestamp']).timestamp() > cutoff_time
                ]
                
                # Make prediction if we have enough data
                if len(self.packet_buffer) >= 10:  # Minimum packets needed
                    predicted_count = self.process_and_predict(self.packet_buffer)
                    
                    if predicted_count is not None:
                        print(f"[{time.strftime('%H:%M:%S')}] Prediction: {predicted_count:.2f} packets")
                        
                        # Check for alerts
                        if self.check_alert(predicted_count):
                            self.send_alert(predicted_count)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Collecting data ({len(self.packet_buffer)} packets)")
        
        except KeyboardInterrupt:
            print("\n\nInference pipeline stopped by user")
        finally:
            self.consumer.close()
            print("Consumer closed")

if __name__ == "__main__":
    # Check if model and scaler exist
    if not os.path.exists('traffic_lstm_model.pth'):
        print("ERROR: Model file not found. Please run train_model.py first.")
        exit(1)
    
    if not os.path.exists('scaler.pkl'):
        print("ERROR: Scaler file not found. Please run train_model.py first.")
        exit(1)
    
    # Initialize and run pipeline
    pipeline = TrafficInferencePipeline(
        alert_threshold=100  # Adjust threshold as needed
    )
    
    pipeline.run(batch_window_seconds=5)

