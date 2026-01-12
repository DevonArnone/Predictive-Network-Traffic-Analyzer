"""
Network Traffic Data Generator
Generates realistic network packet data and streams to Kafka
"""

import json
import time
import random
from datetime import datetime
from confluent_kafka import Producer
from scapy.all import IP, TCP, UDP, ICMP
import socket

class NetworkTrafficGenerator:
    def __init__(self, kafka_bootstrap_servers='localhost:9092', topic='network_traffic'):
        self.producer = Producer({'bootstrap.servers': kafka_bootstrap_servers})
        self.topic = topic
        
        # Common IP ranges for realistic traffic
        self.source_ips = [
            f"192.168.1.{i}" for i in range(1, 255)
        ] + [
            f"10.0.0.{i}" for i in range(1, 100)
        ]
        
        self.dest_ports = [80, 443, 22, 53, 3306, 5432, 8080, 9000, 5000, 3000]
        self.protocols = ['TCP', 'UDP', 'ICMP']
        
    def generate_packet_data(self):
        """Generate a single packet record"""
        protocol = random.choice(self.protocols)
        source_ip = random.choice(self.source_ips)
        dest_port = random.choice(self.dest_ports)
        
        # Realistic packet sizes (Ethernet frame: 64-1500 bytes, IP header: 20 bytes, TCP: 20-60 bytes)
        if protocol == 'TCP':
            packet_length = random.randint(64, 1500)
        elif protocol == 'UDP':
            packet_length = random.randint(28, 1500)
        else:  # ICMP
            packet_length = random.randint(28, 1500)
        
        # Inter-arrival time in milliseconds (simulate realistic network patterns)
        # Mix of bursty and steady traffic
        if random.random() < 0.1:  # 10% chance of burst
            inter_arrival_time = random.uniform(0.1, 1.0)
        else:
            inter_arrival_time = random.uniform(1.0, 100.0)
        
        packet_data = {
            'timestamp': datetime.now().isoformat(),
            'source_ip': source_ip,
            'dest_port': dest_port,
            'protocol': protocol,
            'packet_length': packet_length,
            'inter_arrival_time': round(inter_arrival_time, 3)
        }
        
        return packet_data
    
    def delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err:
            print(f'Error: {err}')
    
    def produce_packet(self, packet_data):
        """Produce a packet to Kafka"""
        try:
            message = json.dumps(packet_data)
            self.producer.produce(
                self.topic,
                message.encode('utf-8'),
                callback=self.delivery_callback
            )
            self.producer.poll(0)  # Trigger delivery callbacks
        except Exception as e:
            print(f'Error producing message: {e}')
    
    def run(self, duration_seconds=300, packets_per_second=10):
        """Run the generator for specified duration"""
        print(f"Generator started - Topic: {self.topic}, Duration: {duration_seconds}s")
        
        start_time = time.time()
        packet_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                packet_data = self.generate_packet_data()
                self.produce_packet(packet_data)
                packet_count += 1
                
                # Sleep based on inter-arrival time
                sleep_time = packet_data['inter_arrival_time'] / 1000.0
                time.sleep(sleep_time)
                
                # Flush producer periodically
                if packet_count % 100 == 0:
                    self.producer.flush()
                    print(f"Produced {packet_count} packets")
            
            # Final flush
            self.producer.flush()
            print(f"Generator finished. Total packets: {packet_count}")
            
        except KeyboardInterrupt:
            print("Generator stopped")
            self.producer.flush()

if __name__ == "__main__":
    generator = NetworkTrafficGenerator()
    # Run for 5 minutes by default, adjust as needed
    generator.run(duration_seconds=300, packets_per_second=10)

