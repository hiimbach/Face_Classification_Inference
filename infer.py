from models.face_classification import FaceClassification
import torch
import numpy as np
from torchvision import transforms

class Face_Classification_Model():
    model = FaceClassification

    def __init__ (self, script_path: str = 'script_models/Face_classifier_Final.pt'):
        # super(Face_Classification_Model, self).__init__()
        # self.model.load_state_dict(torch.load(script_path))
        self.model = torch.jit.load(script_path)
        self.model.eval()

    def predict(self, img:np.ndarray) -> int:

        transform = transforms.ToTensor()

        converted_img = transform(img)
        converted_img.unsqueeze_(dim=0)
    
        yb = self.model(converted_img)  
        
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        return preds[0]
