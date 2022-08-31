from models.face_classification import FaceClassification
import torch

net = FaceClassification()
net.load_state_dict(torch.load('weights/final_model.pth'))

model = torch.jit.script(net)
torch.jit.save(model, "script_models/Face_classifier_Final.pt")