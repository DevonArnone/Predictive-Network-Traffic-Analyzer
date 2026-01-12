"""
Visualize traffic predictions over time
"""

import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
from confluent_kafka import Consumer
from feature_engineering import TrafficFeatureEngineer
from lstm_model import TrafficLSTM
import torch
import numpy as np
import pickle
import os

def collect_predictions(topic='network_traffic', duration=60, prediction_interval=5):
    """Collect predictions from Kafka and return data for visualization"""
    
    if not os.path.exists('traffic_lstm_model.pth'):
        print("ERROR: Model not found. Please train the model first.")
        return None, None
    
    # Load model
    model = TrafficLSTM(input_size=5, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load('traffic_lstm_model.pth', map_location='cpu'))
    model.eval()
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    engineer = TrafficFeatureEngineer(window_size_seconds=5)
    engineer.scaler = scaler
    engineer.is_fitted = True
    
    # Setup consumer
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'visualization_consumer',
        'auto.offset.reset': 'latest'
    })
    consumer.subscribe([topic])
    
    packet_buffer = []
    predictions = []
    timestamps = []
    actual_counts = []
    start_time = time.time()
    last_prediction_time = 0
    
    print(f"Collecting predictions for {duration} seconds...")
    
    try:
        while time.time() - start_time < duration:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                continue
            
            try:
                packet = json.loads(msg.value().decode('utf-8'))
                packet_buffer.append(packet)
                
                # Make prediction at intervals
                if len(packet_buffer) >= 20 and (time.time() - last_prediction_time) >= prediction_interval:
                    try:
                        X = engineer.process_for_inference(packet_buffer, sequence_length=10)
                        X_tensor = torch.FloatTensor(X)
                        
                        with torch.no_grad():
                            prediction = model(X_tensor)
                            predicted_value = prediction.item()
                        
                        # Denormalize
                        dummy = np.zeros((1, 5))
                        dummy[0, 2] = predicted_value
                        denormalized = engineer.scaler.inverse_transform(dummy)
                        predicted_count = max(0, denormalized[0, 2])
                        
                        # Get actual packet count from recent window
                        df = engineer.json_to_dataframe(packet_buffer[-50:])
                        windowed = engineer.create_windows(df)
                        actual_count = windowed['packet_count'].iloc[-1] if len(windowed) > 0 else 0
                        
                        predictions.append(predicted_count)
                        timestamps.append(time.time())
                        actual_counts.append(actual_count)
                        last_prediction_time = time.time()
                        
                        print(f"Time: {time.strftime('%H:%M:%S')} | Predicted: {predicted_count:.2f} | Actual: {actual_count:.2f}")
                        
                        # Keep buffer manageable
                        packet_buffer = packet_buffer[-100:]
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        continue
                        
            except json.JSONDecodeError:
                continue
    
    except KeyboardInterrupt:
        print("\nCollection stopped by user")
    finally:
        consumer.close()
    
    return timestamps, predictions, actual_counts

def plot_results(timestamps, predictions, actual_counts):
    """Create multiple visualizations as separate figures"""
    
    if not timestamps or len(timestamps) == 0:
        print("No data collected. Make sure data generator is running.")
        return
    
    # Convert timestamps to relative time
    start_time = timestamps[0]
    relative_times = [(t - start_time) / 60.0 for t in timestamps]  # Convert to minutes
    threshold = 100
    
    # Figure 1: Prediction vs Actual
    fig1 = plt.figure(1, figsize=(12, 6))
    plt.plot(relative_times, predictions, 'b-', label='Predicted', linewidth=2, marker='o', markersize=4)
    if actual_counts and len(actual_counts) > 0:
        plt.plot(relative_times, actual_counts, 'r--', label='Actual', linewidth=2, marker='s', markersize=4)
    plt.axhline(y=threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold ({threshold})')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Packet Count', fontsize=12)
    plt.title('Figure 1: Predicted vs Actual Traffic', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure1_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    print("Figure 1 saved: figure1_predicted_vs_actual.png")
    
    # Figure 2: Prediction Error
    if actual_counts and len(actual_counts) > 0:
        fig2 = plt.figure(2, figsize=(12, 6))
        errors = [p - a for p, a in zip(predictions, actual_counts)]
        plt.plot(relative_times, errors, 'g-', linewidth=2, marker='o', markersize=4)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Prediction Error', fontsize=12)
        plt.title('Figure 2: Prediction Error Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add error statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=11)
        plt.tight_layout()
        plt.savefig('figure2_prediction_error.png', dpi=150, bbox_inches='tight')
        print("Figure 2 saved: figure2_prediction_error.png")
    
    # Figure 3: Distribution of Predictions
    fig3 = plt.figure(3, figsize=(12, 6))
    plt.hist(predictions, bins=15, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold ({threshold})')
    plt.xlabel('Packet Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Figure 3: Distribution of Predictions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figure3_prediction_distribution.png', dpi=150, bbox_inches='tight')
    print("Figure 3 saved: figure3_prediction_distribution.png")
    
    # Figure 4: Cumulative Traffic
    fig4 = plt.figure(4, figsize=(12, 6))
    cumulative_pred = np.cumsum(predictions)
    plt.plot(relative_times, cumulative_pred, 'b-', label='Cumulative Predicted', linewidth=2)
    if actual_counts and len(actual_counts) > 0:
        cumulative_actual = np.cumsum(actual_counts)
        plt.plot(relative_times, cumulative_actual, 'r--', label='Cumulative Actual', linewidth=2)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Cumulative Packet Count', fontsize=12)
    plt.title('Figure 4: Cumulative Traffic Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure4_cumulative_traffic.png', dpi=150, bbox_inches='tight')
    print("Figure 4 saved: figure4_cumulative_traffic.png")
    
    # Also save combined plot
    fig_combined = plt.figure(5, figsize=(16, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(relative_times, predictions, 'b-', label='Predicted', linewidth=2, marker='o', markersize=3)
    if actual_counts and len(actual_counts) > 0:
        ax1.plot(relative_times, actual_counts, 'r--', label='Actual', linewidth=2, marker='s', markersize=3)
    ax1.axhline(y=threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Time (minutes)', fontsize=11)
    ax1.set_ylabel('Packet Count', fontsize=11)
    ax1.set_title('Predicted vs Actual Traffic', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    if actual_counts and len(actual_counts) > 0:
        ax2 = plt.subplot(2, 2, 2)
        errors = [p - a for p, a in zip(predictions, actual_counts)]
        ax2.plot(relative_times, errors, 'g-', linewidth=2, marker='o', markersize=3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.set_xlabel('Time (minutes)', fontsize=11)
        ax2.set_ylabel('Prediction Error', fontsize=11)
        ax2.set_title('Prediction Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(predictions, bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold ({threshold})')
    ax3.set_xlabel('Packet Count', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Predictions', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(relative_times, cumulative_pred, 'b-', label='Cumulative Predicted', linewidth=2)
    if actual_counts and len(actual_counts) > 0:
        cumulative_actual = np.cumsum(actual_counts)
        ax4.plot(relative_times, cumulative_actual, 'r--', label='Cumulative Actual', linewidth=2)
    ax4.set_xlabel('Time (minutes)', fontsize=11)
    ax4.set_ylabel('Cumulative Packet Count', fontsize=11)
    ax4.set_title('Cumulative Traffic', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('traffic_predictions_combined.png', dpi=150, bbox_inches='tight')
    print("Combined figure saved: traffic_predictions_combined.png")
    
    # Show all plots
    plt.show()

def plot_training_history():
    """Plot training history if available"""
    if not os.path.exists('training_history.pkl'):
        return False
    
    try:
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Model Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = 'training_history.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {output_file}")
        plt.show()
        return True
    except Exception as e:
        print(f"Could not load training history: {e}")
        return False

def main():
    print("=" * 70)
    print("Traffic Prediction Visualization")
    print("=" * 70)
    print()
    
    # Plot training history if available
    if plot_training_history():
        print()
    
    print("Make sure data_generator.py is running in another terminal")
    print("Collecting predictions for 60 seconds...")
    print()
    
    timestamps, predictions, actual_counts = collect_predictions(duration=60, prediction_interval=5)
    
    if timestamps:
        print(f"\nCollected {len(predictions)} predictions")
        plot_results(timestamps, predictions, actual_counts)
    else:
        print("Failed to collect data")

if __name__ == "__main__":
    main()

