import torch
from torchvision.models import resnet18
from torchvision import transforms


class DeepSimilarity(torch.nn.Module):
    def __init__(self):
        super(DeepSimilarity, self).__init__()
        # Image transforms for
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resizing to 224x224 as expected by ResNet
            transforms.ToTensor(),# Convert to tensor [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Norm based on imagenet
        ])

        # Load a ResNet18 model pre-trained on ImageNet
        original_resnet18 = resnet18(pretrained=True)
        # print('---------------')
        # print(*list(original_resnet18.children())[:-1])
        # Remove the last fully connected layer (fc) to use as a feature extractor
        self.feature_extractor = torch.nn.Sequential(*list(original_resnet18.children())[:-3]) 

    def forward(self, x):
        # x shape: (b, 2, c, h, w)
        b, pair, c, h, w = x.shape
        # Flatten the batch and pair dimension for processing through the CNN
        x = x.view(-1, c, h, w)  # Shape: (b*2, c, h, w)
        
        # Get the feature map from the CNN
        
        feature_map = self.feature_extractor(x)  # Shape: (b*2, 256, 14, 14)
       
        feature_map = feature_map.view(b, pair, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])  # Shape: (b,2, 256, 14, 14)
      
        embedding_first = feature_map[:, 0]  # Shape: (b, 256, 14, 14)
        embedding_second = feature_map[:, 1] # Shape: (b, 256, 14, 14)
        #print(embedding_first.shape)
        cosine_similarity = torch.nn.functional.cosine_similarity(embedding_first.flatten(1),embedding_second.flatten(1),dim=1)
        normalised_cosine_similarity = (cosine_similarity + 1) / 2

        normalized_euclidean_similarity =1 / (1 + torch.norm(embedding_first.flatten(1) - embedding_second.flatten(1), p=2, dim=1))
        
        return normalised_cosine_similarity,  normalized_euclidean_similarity


if __name__ == "__main__":
    # Initialize Model
    model = DeepSimilarity()

    # Random test data
    batch_size = 1
    input_data = torch.randn(batch_size, 2, 3, 224, 224)
    # For your input image of arbitrary size, apply model.transform(input_im) to get a formatted Tensor

    # Forward pass
    normalised_cosine_similarity,normalized_eucledian_similarity = model(input_data)
   