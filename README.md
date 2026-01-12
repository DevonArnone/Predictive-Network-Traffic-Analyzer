# Predictive Traffic System

A real-time network traffic prediction system using LSTM neural networks, Kafka streaming, and PyTorch. This system monitors network traffic patterns and predicts potential congestion before it occurs.

## Architecture Overview

1. **Data Generation**: Simulates realistic network packets using Scapy and streams to Kafka
2. **Feature Engineering**: Windows and normalizes raw packet data for LSTM consumption
3. **LSTM Model**: PyTorch-based neural network that learns temporal traffic patterns
4. **Real-Time Inference**: Kafka consumer that processes live traffic and generates predictions
5. **Alerting**: Triggers alerts when predicted traffic exceeds thresholds

## Prerequisites

- Docker and Docker Compose (for Kafka)
- Python 3.8+
- pip

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Kafka Infrastructure

```bash
docker-compose up -d
```

This will start:
- Zookeeper (port 2181)
- Kafka broker (port 9092)

Wait a few seconds for services to initialize. Verify with:
```bash
docker-compose ps
```

### 3. Train the Model

First, you need to generate training data and train the model:

**Terminal 1 - Generate Data:**
```bash
python data_generator.py
```

**Terminal 2 - Train Model:**
```bash
python train_model.py
```

The training script will:
- Collect data from Kafka
- Engineer features (windowing, normalization)
- Train the LSTM model
- Save the model (`traffic_lstm_model.pth`) and scaler (`scaler.pkl`)

**Note**: Let the data generator run for at least 5-10 minutes to collect sufficient training data.

### 4. Run Real-Time Inference

Once the model is trained, start the inference pipeline:

**Terminal 1 - Generate Live Data:**
```bash
python data_generator.py
```

**Terminal 2 - Run Inference:**
```bash
python inference_pipeline.py
```

The inference pipeline will:
- Consume packets from Kafka
- Process them in 5-second windows
- Generate predictions for the next time window
- Alert when traffic exceeds the threshold (default: 100 packets)

## Project Structure

```
.
├── docker-compose.yml          # Kafka infrastructure setup
├── data_generator.py           # Generates mock network packets → Kafka
├── feature_engineering.py      # Windowing, normalization, sequence creation
├── lstm_model.py              # PyTorch LSTM model definition
├── train_model.py             # Training pipeline
├── inference_pipeline.py      # Real-time prediction pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Model Architecture

- **Input**: 5 features per timestep (total_bytes, avg_packet_size, packet_count, avg_inter_arrival_time, protocol_encoded)
- **LSTM Layers**: 2 layers with 64 units each
- **Output**: Predicted packet count for next time window
- **Loss Function**: MSE (Mean Squared Error)
- **Sequence Length**: 10 timesteps (lookback window)

## Configuration

### Adjust Alert Threshold

Edit `inference_pipeline.py`:
```python
pipeline = TrafficInferencePipeline(
    alert_threshold=150  # Change threshold here
)
```

### Adjust Window Size

Edit `feature_engineering.py`:
```python
engineer = TrafficFeatureEngineer(window_size_seconds=10)  # Change window size
```

### Adjust Model Parameters

Edit `lstm_model.py` or `train_model.py`:
```python
model = TrafficLSTM(
    input_size=5,
    hidden_size=128,  # Increase for more capacity
    num_layers=3,      # Add more layers
    output_size=1,
    dropout=0.3
)
```

## Troubleshooting

### Kafka Connection Issues

If you see connection errors:
1. Verify Kafka is running: `docker-compose ps`
2. Check logs: `docker-compose logs kafka`
3. Ensure ports 9092 and 2181 are not in use

### Not Enough Training Data

If training fails with "not enough data":
1. Let the data generator run longer (10+ minutes)
2. Increase `max_messages` in `train_model.py`
3. Reduce `sequence_length` if needed

### Model File Not Found

Ensure you've run `train_model.py` successfully before running inference. The script creates:
- `traffic_lstm_model.pth` (model weights)
- `scaler.pkl` (feature scaler)
- `training_history.pkl` (training metrics)

## Performance Tips

1. **More Training Data**: Longer training data collection improves model accuracy
2. **Hyperparameter Tuning**: Experiment with hidden_size, num_layers, learning_rate
3. **Sequence Length**: Adjust based on your traffic patterns (longer = more context)
4. **Window Size**: Balance between granularity and noise (5-10 seconds recommended)

## Evaluation & Metrics

The project includes scripts to validate performance claims:

### Model Accuracy
```bash
python3 evaluate_model.py
```
Calculates R² score and accuracy metrics on test data.

### Throughput Measurement
```bash
python3 measure_throughput.py
```
Measures Kafka pipeline processing rate (target: 5,000+ messages/second).

### Feature Engineering Impact
```bash
python3 demonstrate_features.py
```
Shows how Pandas/NumPy normalization improves convergence.

### Visualizations
```bash
python3 visualize_predictions.py  # Real-time predictions
python3 visualize_metrics.py       # Performance metrics
```

See `PROJECT_METRICS.md` for detailed validation of all claims.

## Project Structure

```
.
├── data_generator.py          # Generates network packets → Kafka
├── feature_engineering.py     # Windowing, normalization, sequences
├── lstm_model.py              # PyTorch LSTM model definition
├── train_model.py             # Training pipeline
├── inference_pipeline.py      # Real-time prediction pipeline
├── visualize_predictions.py   # Prediction visualizations
├── visualize_metrics.py       # Performance metrics visualizations
├── evaluate_model.py          # Accuracy evaluation
├── measure_throughput.py      # Throughput measurement
├── demonstrate_features.py    # Feature engineering demo
├── docker-compose.yml         # Kafka infrastructure
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── PROJECT_METRICS.md         # Metrics validation
```

## Future Enhancements

- WebSocket-based real-time dashboard
- Multiple prediction targets (latency, bandwidth, etc.)
- Anomaly detection for security threats
- Model retraining pipeline
- Integration with actual network monitoring tools

## License

This project is for educational and demonstration purposes.

