import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.fingergan import UNetGenerator, Discriminator
from models.losses import WeightedReconstructionLoss, StandardReconstructionLoss, AdversarialLoss
from data import FingerprintDataset
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

def train():
    import sys
    # Allow config file selection via command line
    config_path = "configs/train.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    generator = UNetGenerator(**config['model']['generator']).to(device)
    use_discriminator = config['model'].get('use_discriminator', True)
    if use_discriminator:
        discriminator = Discriminator(**config['model']['discriminator']).to(device)
    
    # Loss functions
    use_weighted_loss = config['model'].get('use_weighted_loss', True)
    if use_weighted_loss:
        recon_loss = WeightedReconstructionLoss(config['training']['reconstruction_weight'])
    else:
        recon_loss = StandardReconstructionLoss(config['training']['reconstruction_weight'])
    if use_discriminator:
        adv_loss = AdversarialLoss()
    
    # Optimizers
    g_optim = Adam(generator.parameters(), 
                  lr=config['training']['lr'],
                  betas=(config['training']['beta1'], config['training']['beta2']))
    if use_discriminator:
        d_optim = Adam(discriminator.parameters(),
                      lr=config['training']['lr'],
                      betas=(config['training']['beta1'], config['training']['beta2']))
    
    # Dataset and dataloader
    dataset = FingerprintDataset(config['data']['path'])
    dataloader = DataLoader(dataset, 
                          batch_size=config['data']['batch_size'],
                          num_workers=config['data']['num_workers'],
                          shuffle=config['data']['shuffle'])
    
    # Logger
    writer = SummaryWriter(config['output']['log_dir'])
    
    # Training loop
    use_skeleton_target = config['model'].get('use_skeleton_target', True)
    use_orientation_field = config['model'].get('use_orientation_field', True)
    
    for epoch in range(config['training']['epochs']):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for i, (latent, skeleton, orientation) in enumerate(progress_bar):
            latent = latent.unsqueeze(1).float().to(device)
            skeleton = skeleton.unsqueeze(1).float().to(device)
            orientation = orientation.unsqueeze(1).float().to(device)
            
            # Determine target based on ablation
            if use_skeleton_target:
                target = skeleton
            else:
                target = latent  # No skeleton ablation: latent-to-latent
            
            # Generate weight map from skeleton (simplified)
            if use_weighted_loss:
                weight_map = skeleton * 0.9 + 0.1  # Minutiae areas get weight=1, others 0.1
            else:
                weight_map = torch.ones_like(skeleton)  # No weight ablation: uniform weighting

            if use_discriminator:
                # ---------------------
                # Train Discriminator
                # ---------------------
                d_optim.zero_grad()
                # Real samples
                real_input = torch.cat([skeleton, orientation], dim=1)
                real_pred = discriminator(real_input)
                d_loss_real = adv_loss(real_pred, True)
                # Fake samples
                with torch.no_grad():
                    fake = generator(latent)
                    if fake.shape[2:] != orientation.shape[2:]:
                        fake = torch.nn.functional.interpolate(fake, size=orientation.shape[2:], mode="bilinear", align_corners=False)
                fake_input = torch.cat([fake, orientation], dim=1)
                fake_pred = discriminator(fake_input.detach())
                d_loss_fake = adv_loss(fake_pred, False)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optim.step()

            # -----------------
            # Train Generator
            # -----------------
            g_optim.zero_grad()
            enhanced = generator(latent)
            if enhanced.shape[2:] != target.shape[2:]:
                enhanced = torch.nn.functional.interpolate(enhanced, size=target.shape[2:], mode="bilinear", align_corners=False)
            # Reconstruction loss
            g_loss_rec = recon_loss(enhanced, target, weight_map)
            if use_discriminator:
                # Adversarial loss
                fake_input = torch.cat([enhanced, orientation], dim=1)
                fake_pred = discriminator(fake_input)
                g_loss_adv = adv_loss(fake_pred, True)
                g_loss = g_loss_adv + g_loss_rec
            else:
                g_loss = g_loss_rec
            g_loss.backward()
            g_optim.step()

            # Update progress bar
            if use_discriminator:
                progress_bar.set_postfix({
                    "D Loss": d_loss.item(),
                    "G Loss": g_loss.item(),
                    "Rec": g_loss_rec.item()
                })
            else:
                progress_bar.set_postfix({
                    "G Loss": g_loss.item(),
                    "Rec": g_loss_rec.item()
                })

            # Log to tensorboard
            global_step = epoch * len(dataloader) + i
            if use_discriminator:
                writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
            writer.add_scalar("Loss/Reconstruction", g_loss_rec.item(), global_step)
        # Save models periodically
        if (epoch + 1) % config['output']['save_interval'] == 0:
            os.makedirs(config['output']['save_dir'], exist_ok=True)
            torch.save(generator.state_dict(), 
                      f"{config['output']['save_dir']}/generator_epoch{epoch+1}.pth")
            if use_discriminator:
                torch.save(discriminator.state_dict(),
                          f"{config['output']['save_dir']}/discriminator_epoch{epoch+1}.pth")
    # Save final models
    torch.save(generator.state_dict(), 
              f"{config['output']['save_dir']}/generator_final.pth")
    if use_discriminator:
        torch.save(discriminator.state_dict(),
                  f"{config['output']['save_dir']}/discriminator_final.pth")
    writer.close()

if __name__ == "__main__":
    train()