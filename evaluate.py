import torch
from models.fingergan import UNetGenerator
from data import FingerprintDataset
import matplotlib.pyplot as plt


def evaluate(model_path, data_dir, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Load data
    dataset = FingerprintDataset(data_dir)
    indices = torch.randperm(len(dataset))[:num_samples]
    
    for idx in indices:
        latent, skeleton, _ = dataset[idx]
        latent = latent.unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            enhanced = generator(latent)
            if enhanced.shape[2:] != skeleton.shape[-2:]:
                enhanced = torch.nn.functional.interpolate(enhanced, size=(skeleton.shape[-2], skeleton.shape[-1]), mode="bilinear", align_corners=False)
        
        # Visualize
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(latent.squeeze().cpu(), cmap='gray')
        plt.title("Input Latent")
        
        plt.subplot(1,3,2)
        plt.imshow(enhanced.squeeze().cpu(), cmap='gray')
        plt.title("Enhanced")
        
        plt.subplot(1,3,3)
        plt.imshow(skeleton.squeeze(), cmap='gray')
        plt.title("Ground Truth")
        
        plt.show()

if __name__ == "__main__":
    evaluate("models/generator_final.pth", "data/processed")
