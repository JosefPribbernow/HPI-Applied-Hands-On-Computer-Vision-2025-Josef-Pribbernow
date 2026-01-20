#!/usr/bin/env python3
"""
GPU stress script that performs matrix multiplication on the GPU
every 10 minutes for 30 seconds duration.
"""

import torch
import time
from datetime import datetime

def stress_gpu(duration_seconds=30):
    """Perform intensive matrix multiplication on GPU for specified duration."""
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return False
    
    device = torch.device('cuda')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting GPU stress test for {duration_seconds} seconds...")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create large matrices
    matrix_size = 8192/2  # 4096x4096 matrices
    matrix_size = int(matrix_size)
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration_seconds:
        # Generate random matrices and perform multiplication
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        c = torch.matmul(a, b)
        
        # Force synchronization to ensure GPU work is complete
        torch.cuda.synchronize()
        
        iteration += 1
        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {elapsed:.1f}s / {duration_seconds}s - Iterations: {iteration}")
    
    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed {iteration} matrix multiplications in {total_time:.2f} seconds")
    return True

def main():
    """Main loop that runs GPU stress every 10 minutes."""
    print("GPU Stress Script Started")
    print("Will run matrix multiplication on GPU for 30 seconds every 10 minutes")
    print("Press Ctrl+C to stop\n")
    
    cycle_duration = 10 * 60  # 10 minutes in seconds
    work_duration = 30  # 30 seconds in seconds
    
    try:
        while True:
            # Perform GPU stress
            if not stress_gpu(work_duration):
                print("GPU not available. Exiting.")
                break
            
            # Calculate wait time until next cycle
            wait_time = cycle_duration - work_duration
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sleeping for {wait_time} seconds until next cycle...")
            print(f"Next cycle will start at approximately {datetime.fromtimestamp(time.time() + wait_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting gracefully.")

if __name__ == "__main__":
    main()
