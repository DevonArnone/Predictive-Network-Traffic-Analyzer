"""
Kafka Throughput Measurement
Demonstrates processing rate of 5,000+ data points per second
"""

import json
import time
from confluent_kafka import Producer, Consumer
from datetime import datetime

def measure_producer_throughput(topic='network_traffic', duration=10):
    """Measure producer throughput"""
    producer = Producer({'bootstrap.servers': 'localhost:9092'})
    
    # Generate test packets
    test_packet = {
        'timestamp': datetime.now().isoformat(),
        'source_ip': '192.168.1.1',
        'dest_port': 80,
        'protocol': 'TCP',
        'packet_length': 1500,
        'inter_arrival_time': 0.2
    }
    
    print(f"Measuring producer throughput for {duration} seconds...")
    print("Target: 5,000+ messages per second")
    print()
    
    start_time = time.time()
    message_count = 0
    
    try:
        while time.time() - start_time < duration:
            message = json.dumps(test_packet)
            producer.produce(topic, message.encode('utf-8'))
            message_count += 1
            
            # Poll to trigger delivery
            producer.poll(0)
            
            # Update packet timestamp
            test_packet['timestamp'] = datetime.now().isoformat()
    
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        elapsed = time.time() - start_time
        throughput = message_count / elapsed
    
    return message_count, elapsed, throughput

def measure_consumer_throughput(topic='network_traffic', duration=10):
    """Measure consumer throughput"""
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'throughput_test',
        'auto.offset.reset': 'latest'
    })
    consumer.subscribe([topic])
    
    print(f"Measuring consumer throughput for {duration} seconds...")
    print("(Make sure data_generator.py is running)")
    print()
    
    start_time = time.time()
    message_count = 0
    
    try:
        while time.time() - start_time < duration:
            msg = consumer.poll(timeout=0.1)
            
            if msg is None:
                continue
            if msg.error():
                continue
            
            try:
                json.loads(msg.value().decode('utf-8'))
                message_count += 1
            except:
                continue
    
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        elapsed = time.time() - start_time
        throughput = message_count / elapsed if elapsed > 0 else 0
    
    return message_count, elapsed, throughput

def main():
    print("=" * 70)
    print("Kafka Throughput Measurement")
    print("=" * 70)
    print()
    print("This script measures the data processing rate of the Kafka pipeline")
    print("Target: 5,000+ data points per second")
    print()
    
    # Measure producer throughput
    print("=" * 70)
    print("PRODUCER THROUGHPUT TEST")
    print("=" * 70)
    prod_count, prod_elapsed, prod_throughput = measure_producer_throughput(duration=10)
    
    print(f"Messages produced: {prod_count:,}")
    print(f"Time elapsed: {prod_elapsed:.2f} seconds")
    print(f"Throughput: {prod_throughput:,.2f} messages/second")
    
    if prod_throughput >= 5000:
        print(f"✓ Achieved target: {prod_throughput:,.0f} messages/sec (target: 5,000+)")
    else:
        print(f"Note: {prod_throughput:,.0f} messages/sec (target: 5,000+)")
        print("Throughput may vary based on system resources")
    
    print()
    
    # Measure consumer throughput
    print("=" * 70)
    print("CONSUMER THROUGHPUT TEST")
    print("=" * 70)
    print("Waiting 2 seconds for messages to accumulate...")
    time.sleep(2)
    
    cons_count, cons_elapsed, cons_throughput = measure_consumer_throughput(duration=10)
    
    print(f"Messages consumed: {cons_count:,}")
    print(f"Time elapsed: {cons_elapsed:.2f} seconds")
    print(f"Throughput: {cons_throughput:,.2f} messages/second")
    
    if cons_throughput >= 5000:
        print(f"✓ Achieved target: {cons_throughput:,.0f} messages/sec (target: 5,000+)")
    else:
        print(f"Note: {cons_throughput:,.0f} messages/sec (target: 5,000+)")
        print("Make sure data_generator.py is running to see full throughput")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Producer: {prod_throughput:,.0f} msg/sec")
    print(f"Consumer: {cons_throughput:,.0f} msg/sec")
    print(f"Combined pipeline: {min(prod_throughput, cons_throughput):,.0f} msg/sec")
    print("=" * 70)
    
    # Save results
    results = {
        'producer_throughput': float(prod_throughput),
        'consumer_throughput': float(cons_throughput),
        'producer_messages': int(prod_count),
        'consumer_messages': int(cons_count),
        'target_throughput': 5000
    }
    
    import json as json_lib
    with open('throughput_metrics.json', 'w') as f:
        json_lib.dump(results, f, indent=2)
    
    print("\nThroughput metrics saved to: throughput_metrics.json")

if __name__ == "__main__":
    main()

