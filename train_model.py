"""
Training Script for LSTM Traffic Prediction Model
"""

import json
import numpy as np
from confluent_kafka import Consumer
import torch
from lstm_model import TrafficLSTM, train_model
from feature_engineering import TrafficFeatureEngineer
from sklearn.model_selection import train_test_split

def collect_training_data(kafka_bootstrap_servers='localhost:9092', 
                         topic='network_traffic',
                         duration_seconds=300,
                         max_messages=10000):
    """
    Collect training data from Kafka topic
    
    Args:
        kafka_bootstrap_servers: Kafka broker address
        topic: Kafka topic name
        duration_seconds: How long to collect data
        max_messages: Maximum number of messages to collect
        
    Returns:
        List of packet dictionaries
    """
    consumer = Consumer({
        'bootstrap.servers': kafka_bootstrap_servers,
        'group.id': 'training_consumer',
        'auto.offset.reset': 'earliest'
    })
    
    consumer.subscribe([topic])
    
    print(f"Collecting training data from topic: {topic}")
    print(f"Duration: {duration_seconds} seconds or {max_messages} messages")
    
    packets = []
    message_count = 0
    import time
    start_time = time.time()
    
    try:
        while message_count < max_messages and (time.time() - start_time) < duration_seconds:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            
            try:
                packet = json.loads(msg.value().decode('utf-8'))
                packets.append(packet)
                message_count += 1
                
                if message_count % 100 == 0:
                    print(f"Collected {message_count} packets...")
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    finally:
        consumer.close()
    
    print(f"\nCollected {len(packets)} packets for training")
    return packets

def main():
    """
    Main training pipeline
    """
    print("Training LSTM Traffic Prediction Model")
    print("-" * 60)
    
    # Step 1: Collect training data
    print("Collecting training data from Kafka...")
    training_packets = collect_training_data(duration_seconds=600, max_messages=5000)
    
    if len(training_packets) < 100:
        print("ERROR: Not enough training data. Need at least 100 packets.")
        print("Run data_generator.py first and ensure Kafka is running.")
        return
    
    # Step 2: Feature engineering
    print("Engineering features...")
    engineer = TrafficFeatureEngineer(window_size_seconds=5)
    X, y = engineer.process_for_training(training_packets, sequence_length=10)
    
    print(f"Created {len(X)} sequences (shape: {X.shape})")
    
    if len(X) < 20:
        print("ERROR: Not enough sequences for training.")
        return
    
    # Step 3: Split data
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"Train: {len(X_train)} | Validation: {len(X_val)}")
    
    # Step 4: Initialize model
    print("Initializing model...")
    model = TrafficLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Step 5: Train model
    print("Training...")
    trained_model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Step 6: Save model and scaler
    print("Saving model...")
    torch.save(trained_model.state_dict(), 'traffic_lstm_model.pth')
    
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(engineer.scaler, f)
    
    print("Saved: traffic_lstm_model.pth, scaler.pkl")
    
    # Step 7: Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("Training completed")

if __name__ == "__main__":
    main()

