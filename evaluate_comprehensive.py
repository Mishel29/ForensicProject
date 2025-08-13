import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.fingergan import UNetGenerator
from data import FingerprintDataset
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_model(model_path, config):
    """Load a trained generator model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetGenerator(**config['model']['generator']).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint)
        generator.eval()
        return generator, device
    else:
        print(f"Warning: Model not found at {model_path}")
        return None, device

def evaluate_model(generator, dataloader, device, model_name, config, num_samples=20):
    """Evaluate model on test samples"""
    if generator is None:
        return None
        
    results = {
        'model_name': model_name,
        'psnr_scores': [],
        'ssim_scores': [],
        'mse_scores': [],
        'sample_images': [],
        'sample_targets': [],
        'sample_outputs': []
    }
    
    # Configuration options
    use_skeleton_target = config['model'].get('use_skeleton_target', True)
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, (latent, skeleton, orientation) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            latent = latent.unsqueeze(1).float().to(device)
            skeleton = skeleton.unsqueeze(1).float().to(device)
            orientation = orientation.unsqueeze(1).float().to(device)
            
            # Determine target based on configuration
            if use_skeleton_target:
                target = skeleton
            else:
                target = latent  # For no-skeleton ablation
            
            # Generate enhanced images
            enhanced = generator(latent)
            if enhanced.shape[2:] != target.shape[2:]:
                enhanced = torch.nn.functional.interpolate(enhanced, size=target.shape[2:], mode="bilinear", align_corners=False)
            
            # Convert to numpy for metrics calculation
            for i in range(latent.size(0)):
                if sample_count >= num_samples:
                    break
                    
                latent_np = latent[i].cpu().numpy().squeeze()
                target_np = target[i].cpu().numpy().squeeze()
                enhanced_np = enhanced[i].cpu().numpy().squeeze()
                
                # Calculate metrics
                psnr_score = psnr(target_np, enhanced_np, data_range=1.0)
                ssim_score = ssim(target_np, enhanced_np, data_range=1.0)
                mse_score = np.mean((target_np - enhanced_np) ** 2)
                
                results['psnr_scores'].append(psnr_score)
                results['ssim_scores'].append(ssim_score)
                results['mse_scores'].append(mse_score)
                
                # Store sample images for visualization
                if len(results['sample_images']) < 3:  # Store first 3 samples
                    results['sample_images'].append(latent_np)
                    results['sample_targets'].append(target_np)
                    results['sample_outputs'].append(enhanced_np)
                
                sample_count += 1
    
    # Calculate average metrics
    if results['psnr_scores']:
        results['avg_psnr'] = np.mean(results['psnr_scores'])
        results['avg_ssim'] = np.mean(results['ssim_scores'])
        results['avg_mse'] = np.mean(results['mse_scores'])
        results['std_psnr'] = np.std(results['psnr_scores'])
        results['std_ssim'] = np.std(results['ssim_scores'])
        results['std_mse'] = np.std(results['mse_scores'])
    
    return results

def create_comprehensive_comparison(all_results, save_dir):
    """Create comprehensive comparison visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out None results
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    # 1. Comprehensive metrics comparison
    plt.figure(figsize=(18, 6))
    
    models = list(valid_results.keys())
    
    # PSNR comparison
    plt.subplot(1, 3, 1)
    psnr_means = [valid_results[model]['avg_psnr'] for model in models]
    psnr_stds = [valid_results[model]['std_psnr'] for model in models]
    
    bars = plt.bar(range(len(models)), psnr_means, yerr=psnr_stds, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.7)
    plt.title('PSNR Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('PSNR (dB)')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, psnr_means, psnr_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM comparison
    plt.subplot(1, 3, 2)
    ssim_means = [valid_results[model]['avg_ssim'] for model in models]
    ssim_stds = [valid_results[model]['std_ssim'] for model in models]
    
    bars = plt.bar(range(len(models)), ssim_means, yerr=ssim_stds, capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.7)
    plt.title('SSIM Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('SSIM')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    
    for i, (bar, mean, std) in enumerate(zip(bars, ssim_means, ssim_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison
    plt.subplot(1, 3, 3)
    mse_means = [valid_results[model]['avg_mse'] for model in models]
    mse_stds = [valid_results[model]['std_mse'] for model in models]
    
    bars = plt.bar(range(len(models)), mse_means, yerr=mse_stds, capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.7)
    plt.title('MSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('MSE')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    
    for i, (bar, mean, std) in enumerate(zip(bars, mse_means, mse_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.002,
                f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample outputs comparison
    num_models = len(valid_results)
    fig, axes = plt.subplots(num_models + 1, 3, figsize=(12, 4 * (num_models + 1)))
    
    # Get a sample from the first model for input/target
    first_model = list(valid_results.keys())[0]
    sample_input = valid_results[first_model]['sample_images'][0]
    sample_target = valid_results[first_model]['sample_targets'][0]
    
    # Show input and target
    axes[0, 0].imshow(sample_input, cmap='gray')
    axes[0, 0].set_title('Input (Latent)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sample_target, cmap='gray')
    axes[0, 1].set_title('Target', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')  # Empty cell
    
    # Show outputs from each model
    for i, (model_name, results) in enumerate(valid_results.items()):
        row = i + 1
        
        # Model output
        if results['sample_outputs']:
            axes[row, 0].imshow(results['sample_outputs'][0], cmap='gray')
            axes[row, 0].set_title(f'{model_name} Output', fontweight='bold')
            axes[row, 0].axis('off')
            
            # Difference map
            diff = np.abs(sample_target - results['sample_outputs'][0])
            im = axes[row, 1].imshow(diff, cmap='hot')
            axes[row, 1].set_title(f'{model_name} Error Map', fontweight='bold')
            axes[row, 1].axis('off')
            
            # Metrics
            axes[row, 2].text(0.1, 0.8, f"PSNR: {results['avg_psnr']:.2f} dB", fontsize=12, transform=axes[row, 2].transAxes)
            axes[row, 2].text(0.1, 0.6, f"SSIM: {results['avg_ssim']:.3f}", fontsize=12, transform=axes[row, 2].transAxes)
            axes[row, 2].text(0.1, 0.4, f"MSE: {results['avg_mse']:.4f}", fontsize=12, transform=axes[row, 2].transAxes)
            axes[row, 2].axis('off')
    
    plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run comprehensive evaluation of all ablation studies"""
    print("=== COMPREHENSIVE ABLATION EVALUATION ===\n")
    
    # Define all models to evaluate
    models_config = {
        "Full FingerGAN": {
            "config": "configs/train.yaml",
            "model": "models/generator_final.pth"
        },
        "No Discriminator": {
            "config": "configs/ablation_no_discriminator.yaml", 
            "model": "models/ablation_no_discriminator/generator_final.pth"
        },
        "No Weight": {
            "config": "configs/ablation_no_weight.yaml",
            "model": "models/ablation_no_weight/generator_final.pth"
        },
        "No Skeleton": {
            "config": "configs/ablation_no_skeleton.yaml",
            "model": "models/ablation_no_skeleton/generator_final.pth"
        },
        "No Orientation": {
            "config": "configs/ablation_no_orientation.yaml",
            "model": "models/ablation_no_orientation/generator_final.pth"
        }
    }
    
    # Load base config for dataset
    with open('configs/train.yaml') as f:
        base_config = yaml.safe_load(f)
    
    # Create test dataset
    test_dataset = FingerprintDataset(base_config['data']['path'])
    test_size = len(test_dataset) // 5  # 20% for testing
    test_indices = list(range(len(test_dataset) - test_size, len(test_dataset)))
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    test_dataloader = DataLoader(
        test_subset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Loaded test dataset with {len(test_subset)} samples\n")
    
    # Evaluate all models
    all_results = {}
    
    for model_name, model_info in models_config.items():
        print(f"🔬 Evaluating {model_name}...")
        
        # Load model config
        try:
            with open(model_info['config']) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"   ❌ Config not found: {model_info['config']}")
            all_results[model_name] = None
            continue
        
        # Load and evaluate model
        generator, device = load_model(model_info['model'], config)
        results = evaluate_model(generator, test_dataloader, device, model_name, config)
        all_results[model_name] = results
        
        if results:
            print(f"   ✅ PSNR: {results['avg_psnr']:.2f} dB, SSIM: {results['avg_ssim']:.3f}, MSE: {results['avg_mse']:.4f}")
        else:
            print(f"   ❌ Evaluation failed")
    
    # Create comprehensive results table
    print(f"\n{'='*80}")
    print("🏆 COMPREHENSIVE ABLATION RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Model':<20} {'PSNR (dB)':<12} {'SSIM':<12} {'MSE':<12} {'Status'}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        if results:
            psnr_val = f"{results['avg_psnr']:.2f}"
            ssim_val = f"{results['avg_ssim']:.3f}"
            mse_val = f"{results['avg_mse']:.4f}"
            status = "✅"
        else:
            psnr_val = "N/A"
            ssim_val = "N/A" 
            mse_val = "N/A"
            status = "❌"
        
        print(f"{model_name:<20} {psnr_val:<12} {ssim_val:<12} {mse_val:<12} {status}")
    
    # Find best performing model
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        best_psnr = max(valid_results.items(), key=lambda x: x[1]['avg_psnr'])
        best_ssim = max(valid_results.items(), key=lambda x: x[1]['avg_ssim'])
        best_mse = min(valid_results.items(), key=lambda x: x[1]['avg_mse'])
        
        print(f"\n🥇 BEST PERFORMERS:")
        print(f"   PSNR: {best_psnr[0]} ({best_psnr[1]['avg_psnr']:.2f} dB)")
        print(f"   SSIM: {best_ssim[0]} ({best_ssim[1]['avg_ssim']:.3f})")
        print(f"   MSE:  {best_mse[0]} ({best_mse[1]['avg_mse']:.4f})")
    
    # Create visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    create_comprehensive_comparison(all_results, 'comprehensive_evaluation_results')
    
    print(f"\n✅ Comprehensive evaluation complete!")
    print(f"   Check 'comprehensive_evaluation_results/' for detailed analysis")
    print(f"   - comprehensive_metrics_comparison.png")
    print(f"   - comprehensive_model_comparison.png")

if __name__ == "__main__":
    main()
