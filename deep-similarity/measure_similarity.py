import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from image_dataloader import ImageDataSet
from model import DeepSimilarity

def calculate_similarities(dataloader, model, device):
    model.eval()
    cosine_similarities, euclidean_similarities = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            cosine_sim, euclidean_sim = model(batch)
            cosine_similarities.extend(cosine_sim.cpu().numpy())
            euclidean_similarities.extend(euclidean_sim.cpu().numpy())
    return np.mean(cosine_similarities), np.mean(euclidean_similarities)

def main():
    parser = argparse.ArgumentParser(description="Calculate average similarity between image pairs")
    parser.add_argument('--pair', type=str, default='both', choices=['stereo', 'adjacent', 'both'], help="Data loading pair")
    parser.add_argument('--data_path', type=str, required=True, help="Root directory of the dataset")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSimilarity().to(device)

    results = {}

    if args.pair in ['stereo', 'both']:
        stereo_dataset = ImageDataSet(data_path=args.data_path, pair='stereo',transform=model.transform)
        stereo_loader = DataLoader(stereo_dataset, batch_size=args.batch_size, shuffle=False)
        stereo_cosine, stereo_euclidean = calculate_similarities(stereo_loader, model, device)
        results['stereo'] = (stereo_cosine, stereo_euclidean)

    if args.pair in ['adjacent', 'both']:
        mono_dataset =ImageDataSet(data_path=args.data_path, pair='adjacent',transform=model.transform)
        mono_loader = DataLoader(mono_dataset, batch_size=args.batch_size, shuffle=False)
        mono_cosine, mono_euclidean = calculate_similarities(mono_loader, model, device)
        results['adjacent'] = (mono_cosine, mono_euclidean)

    for pair, (cos_sim, euc_dist) in results.items():
        print(f"{pair.capitalize()} - Avg Cosine Similarity: {cos_sim:.4f}, Avg Euclidean Similarity: {euc_dist:.4f}")

if __name__ == "__main__":
    main()
