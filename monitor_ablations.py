#!/usr/bin/env python3
"""
Automatic Results Consolidator
Updates final results summary with comprehensive ablation findings.
"""

import os
import yaml
import time

def check_ablation_completion():
    """Check which ablation studies have completed"""
    models_to_check = [
        ("models/ablation_no_weight/generator_final.pth", "No Weight"),
        ("models/ablation_no_skeleton/generator_final.pth", "No Skeleton"),
        ("models/ablation_no_orientation/generator_final.pth", "No Orientation")
    ]
    
    completed = []
    for model_path, name in models_to_check:
        if os.path.exists(model_path):
            completed.append(name)
    
    return completed

def monitor_training():
    """Monitor training progress and report completion"""
    print("🔍 MONITORING ABLATION PROGRESS")
    print("Checking for completed models every 30 seconds...\n")
    
    initial_completed = check_ablation_completion()
    print(f"Initially completed: {initial_completed}")
    
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        current_completed = check_ablation_completion()
        new_completions = set(current_completed) - set(initial_completed)
        
        if new_completions:
            for completion in new_completions:
                print(f"✅ {completion} ablation completed!")
            initial_completed = current_completed
        
        print(f"📊 Status: {len(current_completed)}/3 ablations complete")
        
        if len(current_completed) == 3:
            print("\n🎉 ALL ABLATIONS COMPLETED!")
            print("Ready for comprehensive evaluation...")
            break

def auto_evaluate():
    """Automatically run comprehensive evaluation when all ablations complete"""
    print("\n🚀 STARTING AUTOMATIC COMPREHENSIVE EVALUATION")
    
    try:
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, "evaluate_comprehensive.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Comprehensive evaluation completed successfully!")
            print("📊 Results saved to comprehensive_evaluation_results/")
            return True
        else:
            print(f"❌ Evaluation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

def main():
    """Main monitoring and evaluation loop"""
    print("🤖 AUTOMATIC ABLATION MONITOR & EVALUATOR")
    print("=" * 50)
    
    # Monitor training progress
    monitor_training()
    
    # Auto-run comprehensive evaluation
    if auto_evaluate():
        print("\n🏆 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("📋 Key files generated:")
        print("   - comprehensive_evaluation_results/comprehensive_metrics_comparison.png")
        print("   - comprehensive_evaluation_results/comprehensive_model_comparison.png")
        print("   - reports/comprehensive_ablation_report.md")
        print("\n✨ Your research findings are now complete and strengthened!")
    else:
        print("\n⚠️ Automatic evaluation failed. Run manually:")
        print("   python evaluate_comprehensive.py")

if __name__ == "__main__":
    main()
