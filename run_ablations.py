#!/usr/bin/env python3
"""
Comprehensive Ablation Study Runner
Executes all remaining ablation studies to strengthen research findings.
"""

import subprocess
import sys
import time
import os

def run_ablation(config_name, description):
    """Run a single ablation study"""
    print(f"\n{'='*60}")
    print(f"🔬 STARTING ABLATION: {description}")
    print(f"   Config: {config_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run training
        cmd = [sys.executable, "train.py", f"configs/{config_name}"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        duration = time.time() - start_time
        print(f"\n✅ {description} COMPLETED in {duration:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} FAILED after {duration:.1f}s")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run comprehensive ablation study suite"""
    print("🚀 COMPREHENSIVE ABLATION STUDY SUITE")
    print("Strengthening research findings with systematic component analysis\n")
    
    # Define ablation studies
    ablations = [
        ("ablation_no_weight.yaml", "No Minutia Weight (Standard L1 Loss)"),
        ("ablation_no_skeleton.yaml", "No Skeleton Map (Latent-to-Latent)"), 
        ("ablation_no_orientation.yaml", "No Orientation Field (Skip Orientation)")
    ]
    
    results = {}
    total_start = time.time()
    
    # Run each ablation
    for config, description in ablations:
        success = run_ablation(config, description)
        results[description] = success
        
        if success:
            print(f"   📊 Model saved to: models/{config.replace('.yaml', '').replace('ablation_', 'ablation_')}/")
            print(f"   📈 Logs saved to: logs/{config.replace('.yaml', '').replace('ablation_', 'ablation_')}/")
    
    # Summary
    total_duration = time.time() - total_start
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"🏁 ABLATION SUITE COMPLETED")
    print(f"   Total Time: {total_duration:.1f}s")
    print(f"   Success Rate: {successful}/{total} ablations")
    print(f"{'='*60}")
    
    print(f"\n📋 RESULTS SUMMARY:")
    for description, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status}: {description}")
    
    if successful == total:
        print(f"\n🎉 ALL ABLATIONS COMPLETED SUCCESSFULLY!")
        print(f"   Ready for comprehensive evaluation and analysis.")
        print(f"   Next step: Run evaluation script to compare all models.")
    else:
        print(f"\n⚠️  Some ablations failed. Check error messages above.")
    
    print(f"\n📁 Generated Models:")
    models_dir = "models/"
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            if item.startswith("ablation_"):
                print(f"   - {item}/")

if __name__ == "__main__":
    main()
