import time
import os
import csv
import psutil
import numpy as np
from contextlib import contextmanager

class PerformanceMetrics:
    """Class to track and analyze performance metrics for NSGA-III implementations"""
    
    def __init__(self):
        self.metrics = {}
        self.baseline_time = None
        
    def record_implementation(self, name, time_ms, memory_mb=0, throughput=0, 
                             transfer_time=0, kernel_time=0, energy=0):
        """Record metrics for a specific implementation"""
        self.metrics[name] = {
            "time_ms": time_ms,
            "memory_mb": memory_mb,
            "throughput": throughput,
            "memory_transfer_ms": transfer_time,
            "kernel_time_ms": kernel_time,
            "energy_usage_w": energy
        }
        
        # Set as baseline if it's the first implementation or specified as CPU Sequential
        if self.baseline_time is None or name == "CPU Sequential":
            self.baseline_time = time_ms
            
        # Calculate speedup
        if self.baseline_time > 0:
            self.metrics[name]["speedup"] = self.baseline_time / time_ms
        else:
            self.metrics[name]["speedup"] = 1.0
            
    def export_to_csv(self, filename="nsga_performance.csv"):
        """Export performance metrics to CSV file"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Prepare data for CSV
        headers = ["Implementation", "Time (ms)", "Speedup", "Memory Usage (MB)", 
                  "Throughput (evals/s)", "Memory Transfer (ms)", "Kernel Time (ms)", 
                  "Energy Usage (W)"]
        
        rows = []
        for impl_name, data in self.metrics.items():
            rows.append([
                impl_name,
                round(data["time_ms"], 2),
                f"{data['speedup']:.2f}x",
                round(data["memory_mb"], 2),
                round(data["throughput"], 2),
                round(data["memory_transfer_ms"], 2) if data["memory_transfer_ms"] > 0 else "N/A",
                round(data["kernel_time_ms"], 2) if data["kernel_time_ms"] > 0 else "N/A",
                round(data["energy_usage_w"], 2) if data["energy_usage_w"] > 0 else "N/A"
            ])
        
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"Performance metrics exported to {filename}")
    
    def print_comparison(self):
        """Print performance comparison to console"""
        print("\n==== NSGA-III Performance Comparison ====")
        print(f"{'Implementation':<20} | {'Time (ms)':<10} | {'Speedup':<10} | {'Memory (MB)':<12} | {'Throughput':<15}")
        print("-" * 75)
        
        for impl_name, data in self.metrics.items():
            print(f"{impl_name:<20} | {data['time_ms']:<10.2f} | {data['speedup']:<10.2f}x | "
                  f"{data['memory_mb']:<12.2f} | {data['throughput']:<15.2f}")