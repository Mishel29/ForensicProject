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
import cv2

def load_model(model_path, config):
    """Load a trained generator model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetGenerator(**config['model']['generator']).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()
    return generator, device

def evaluate_model(generator, dataloader, device, model_name, num_samples=20):
    """Evaluate model on test samples"""
    results = {
        'model_name': model_name,
        'psnr_scores': [],
        'ssim_scores': [],
        'mse_scores': [],
        'sample_images': [],
        'sample_targets': [],
        'sample_outputs': []
    }
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, (latent, skeleton, orientation) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            latent = latent.unsqueeze(1).float().to(device)
            skeleton = skeleton.unsqueeze(1).float().to(device)
            orientation = orientation.unsqueeze(1).float().to(device)
            
            # Generate enhanced images
            enhanced = generator(latent)
            if enhanced.shape[2:] != skeleton.shape[2:]:
                enhanced = torch.nn.functional.interpolate(enhanced, size=skeleton.shape[2:], mode="bilinear", align_corners=False)
            
            # Convert to numpy for metrics calculation
            for i in range(latent.size(0)):
                if sample_count >= num_samples:
                    break
                    
                latent_np = latent[i].cpu().numpy().squeeze()
                skeleton_np = skeleton[i].cpu().numpy().squeeze()  # This is our target
                enhanced_np = enhanced[i].cpu().numpy().squeeze()
                
                # Calculate metrics (comparing enhanced output to skeleton target)
                psnr_score = psnr(skeleton_np, enhanced_np, data_range=1.0)
                ssim_score = ssim(skeleton_np, enhanced_np, data_range=1.0)
                mse_score = np.mean((skeleton_np - enhanced_np) ** 2)
                
                results['psnr_scores'].append(psnr_score)
                results['ssim_scores'].append(ssim_score)
                results['mse_scores'].append(mse_score)
                
                # Store sample images for visualization
                if len(results['sample_images']) < 5:  # Store first 5 samples
                    results['sample_images'].append(latent_np)
                    results['sample_targets'].append(skeleton_np)
                    results['sample_outputs'].append(enhanced_np)
                
                sample_count += 1
    
    # Calculate average metrics
    results['avg_psnr'] = np.mean(results['psnr_scores'])
    results['avg_ssim'] = np.mean(results['ssim_scores'])
    results['avg_mse'] = np.mean(results['mse_scores'])
    results['std_psnr'] = np.std(results['psnr_scores'])
    results['std_ssim'] = np.std(results['ssim_scores'])
    results['std_mse'] = np.std(results['mse_scores'])
    
    return results

def create_comparison_visualization(full_gan_results, ablation_results, save_dir):
    """Create comparison visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Metrics comparison bar chart
    plt.figure(figsize=(15, 5))
    
    # PSNR comparison
    plt.subplot(1, 3, 1)
    models = ['Full FingerGAN', 'No Discriminator']
    psnr_means = [full_gan_results['avg_psnr'], ablation_results['avg_psnr']]
    psnr_stds = [full_gan_results['std_psnr'], ablation_results['std_psnr']]
    
    bars = plt.bar(models, psnr_means, yerr=psnr_stds, capsize=5, 
                   color=['#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.title('PSNR Comparison')
    plt.ylabel('PSNR (dB)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, psnr_means, psnr_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM comparison
    plt.subplot(1, 3, 2)
    ssim_means = [full_gan_results['avg_ssim'], ablation_results['avg_ssim']]
    ssim_stds = [full_gan_results['std_ssim'], ablation_results['std_ssim']]
    
    bars = plt.bar(models, ssim_means, yerr=ssim_stds, capsize=5,
                   color=['#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.title('SSIM Comparison')
    plt.ylabel('SSIM')
    plt.xticks(rotation=45)
    
    for i, (bar, mean, std) in enumerate(zip(bars, ssim_means, ssim_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison (lower is better)
    plt.subplot(1, 3, 3)
    mse_means = [full_gan_results['avg_mse'], ablation_results['avg_mse']]
    mse_stds = [full_gan_results['std_mse'], ablation_results['std_mse']]
    
    bars = plt.bar(models, mse_means, yerr=mse_stds, capsize=5,
                   color=['#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.title('MSE Comparison (Lower is Better)')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    
    for i, (bar, mean, std) in enumerate(zip(bars, mse_means, mse_stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.00001,
                f'{mean:.6f}±{std:.6f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample images comparison
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i in range(5):  # Show 5 sample images
        # Input (latent) image
        axes[0, i].imshow(full_gan_results['sample_images'][i], cmap='gray')
        axes[0, i].set_title(f'Latent Input {i+1}')
        axes[0, i].axis('off')
        
        # Full FingerGAN output
        axes[1, i].imshow(full_gan_results['sample_outputs'][i], cmap='gray')
        axes[1, i].set_title(f'Full GAN Output {i+1}')
        axes[1, i].axis('off')
        
        # No Discriminator output
        axes[2, i].imshow(ablation_results['sample_outputs'][i], cmap='gray')
        axes[2, i].set_title(f'No Discriminator Output {i+1}')
        axes[2, i].axis('off')
    
    plt.suptitle('Sample Outputs Comparison\\n(Target: Skeleton Structure)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_outputs_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Target comparison (show what models should generate)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(5):
        # Input (latent) image
        axes[0, i].imshow(full_gan_results['sample_images'][i], cmap='gray')
        axes[0, i].set_title(f'Latent Input {i+1}')
        axes[0, i].axis('off')
        
        # Target (skeleton) image
        axes[1, i].imshow(full_gan_results['sample_targets'][i], cmap='gray')
        axes[1, i].set_title(f'Target (Skeleton) {i+1}')
        axes[1, i].axis('off')
        
        # Full FingerGAN output
        axes[2, i].imshow(full_gan_results['sample_outputs'][i], cmap='gray')
        axes[2, i].set_title(f'Full GAN Output {i+1}')
        axes[2, i].axis('off')
        
        # No Discriminator output
        axes[3, i].imshow(ablation_results['sample_outputs'][i], cmap='gray')
        axes[3, i].set_title(f'No Discriminator Output {i+1}')
        axes[3, i].axis('off')
    
    plt.suptitle('Complete Enhancement Pipeline Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complete_pipeline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Difference maps
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i in range(5):
        target = full_gan_results['sample_targets'][i]
        full_gan_output = full_gan_results['sample_outputs'][i]
        ablation_output = ablation_results['sample_outputs'][i]
        
        # Difference map for Full GAN
        diff_full = np.abs(target - full_gan_output)
        axes[0, i].imshow(diff_full, cmap='hot')
        axes[0, i].set_title(f'Full GAN Diff {i+1}\nMSE: {np.mean(diff_full**2):.6f}')
        axes[0, i].axis('off')
        
        # Difference map for Ablation
        diff_ablation = np.abs(target - ablation_output)
        axes[1, i].imshow(diff_ablation, cmap='hot')
        axes[1, i].set_title(f'No Discriminator Diff {i+1}\nMSE: {np.mean(diff_ablation**2):.6f}')
        axes[1, i].axis('off')
    
    plt.suptitle('Error Difference Maps (Brighter = Higher Error)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'difference_maps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== FingerGAN Ablation Evaluation ===\n")
    
    # Load configurations
    with open('configs/train.yaml') as f:
        full_config = yaml.safe_load(f)
    
    with open('configs/ablation_no_discriminator.yaml') as f:
        ablation_config = yaml.safe_load(f)
    
    # Create test dataset using existing data
    test_dataset = FingerprintDataset(full_config['data']['path'])
    
    # Use a subset for evaluation (last 20% of data as test set)
    test_size = len(test_dataset) // 5  # 20% for testing
    test_indices = list(range(len(test_dataset) - test_size, len(test_dataset)))
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    test_dataloader = DataLoader(
        test_subset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Loaded test dataset with {len(test_subset)} samples")
    
    # Evaluate Full FingerGAN
    print("\n1. Evaluating Full FingerGAN...")
    full_gan_generator, device = load_model('models/generator_final.pth', full_config)
    full_gan_results = evaluate_model(full_gan_generator, test_dataloader, device, 'Full FingerGAN')
    
    # Evaluate No Discriminator Ablation
    print("2. Evaluating No Discriminator Ablation...")
    ablation_generator, device = load_model('models/ablation_no_discriminator/generator_final.pth', ablation_config)
    ablation_results = evaluate_model(ablation_generator, test_dataloader, device, 'No Discriminator')
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"\nFull FingerGAN:")
    print(f"  PSNR: {full_gan_results['avg_psnr']:.2f} ± {full_gan_results['std_psnr']:.2f} dB")
    print(f"  SSIM: {full_gan_results['avg_ssim']:.3f} ± {full_gan_results['std_ssim']:.3f}")
    print(f"  MSE:  {full_gan_results['avg_mse']:.6f} ± {full_gan_results['std_mse']:.6f}")
    
    print(f"\nNo Discriminator Ablation:")
    print(f"  PSNR: {ablation_results['avg_psnr']:.2f} ± {ablation_results['std_psnr']:.2f} dB")
    print(f"  SSIM: {ablation_results['avg_ssim']:.3f} ± {ablation_results['std_ssim']:.3f}")
    print(f"  MSE:  {ablation_results['avg_mse']:.6f} ± {ablation_results['std_mse']:.6f}")
    
    # Calculate improvements
    psnr_improvement = ablation_results['avg_psnr'] - full_gan_results['avg_psnr']
    ssim_improvement = ablation_results['avg_ssim'] - full_gan_results['avg_ssim']
    mse_improvement = full_gan_results['avg_mse'] - ablation_results['avg_mse']  # Lower MSE is better
    
    print(f"\n=== COMPARISON ===")
    print(f"PSNR Improvement: {psnr_improvement:+.2f} dB ({100*psnr_improvement/full_gan_results['avg_psnr']:+.1f}%)")
    print(f"SSIM Improvement: {ssim_improvement:+.3f} ({100*ssim_improvement/full_gan_results['avg_ssim']:+.1f}%)")
    print(f"MSE Improvement:  {mse_improvement:+.6f} ({100*mse_improvement/full_gan_results['avg_mse']:+.1f}%)")
    
    # Determine winner
    print(f"\n=== WINNER ===")
    metrics_won = 0
    if psnr_improvement > 0:
        print("✅ PSNR: No Discriminator wins")
        metrics_won += 1
    else:
        print("❌ PSNR: Full FingerGAN wins")
    
    if ssim_improvement > 0:
        print("✅ SSIM: No Discriminator wins")
        metrics_won += 1
    else:
        print("❌ SSIM: Full FingerGAN wins")
    
    if mse_improvement > 0:
        print("✅ MSE: No Discriminator wins")
        metrics_won += 1
    else:
        print("❌ MSE: Full FingerGAN wins")
    
    if metrics_won >= 2:
        print(f"\n🏆 OVERALL WINNER: No Discriminator Ablation ({metrics_won}/3 metrics)")
    else:
        print(f"\n🏆 OVERALL WINNER: Full FingerGAN ({3-metrics_won}/3 metrics)")
    
    # Create visualizations
    print("\n3. Creating comparison visualizations...")
    create_comparison_visualization(full_gan_results, ablation_results, 'evaluation_results')
    
    print("\n✅ Evaluation complete! Check 'evaluation_results/' for visualizations.")
    print("   - metrics_comparison.png: Quantitative metrics comparison")
    print("   - sample_outputs_comparison.png: Side-by-side output comparison")
    print("   - complete_pipeline_comparison.png: Full pipeline visualization")
    print("   - difference_maps.png: Error visualization")

if __name__ == "__main__":
    main()
