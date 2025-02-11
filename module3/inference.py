import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from engine import build_model

class ObjectCounter:
    def __init__(self, checkpoint_path, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(args)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path, bboxes):
        """
        Predict object count for a single image.
        
        Args:
            image_path: Path to the input image
            bboxes: Exemplar bounding boxes [K, 4]
            
        Returns:
            count: Predicted object count
            density_map: Generated density map
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Process bboxes
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes = bboxes.unsqueeze(0).to(self.device)
        
        # Model inference
        with torch.cuda.amp.autocast():
            density_map, _ = self.model(image, bboxes)
        
        # Calculate count
        count = density_map.sum().item()
        
        return count, density_map.cpu().numpy()

def main(args):
    counter = ObjectCounter(args.checkpoint_path, args)
    
    # Example usage
    image_path = args.image_path
    bboxes = np.load(args.bbox_path)  # Load exemplar bboxes
    
    count, density_map = counter.predict(image_path, bboxes)
    print(f"Predicted count: {count:.2f}")
    
    # Optionally save density map visualization
    if args.save_visualization:
        import matplotlib.pyplot as plt
        plt.imshow(density_map[0, 0], cmap='jet')
        plt.colorbar()
        plt.savefig(args.output_path)
        plt.close()

if __name__ == "__main__":
    from arg_parser import get_argparser
    parser = get_argparser()
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--bbox_path', type=str, required=True,
                       help='Path to exemplar bounding boxes')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save density map visualization')
    parser.add_argument('--output_path', type=str, default='density_map.png',
                       help='Path to save visualization')
    args = parser.parse_args()
    main(args)