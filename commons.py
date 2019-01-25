import io

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def get_model():
  checkpoint = 'vgg19_classifier.pth'
  
  model = models.vgg19(pretrained=True)
  for param in model.parameters():
      param.requires_grad = False
            
  classifier = nn.Sequential(nn.Linear(25088, 4096), 
                         nn.ReLU(),
                         nn.Dropout(p=0.1),
                         nn.Linear(4096, 512),
                         nn.ReLU(),
                         nn.Dropout(p=0.1),
                         nn.Linear(512, 102),                  
                         nn.LogSoftmax(dim=1))
  
  model.classifier = classifier
  model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)
  model.eval()
  return model


def get_tensor(image_bytes):
  my_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
  image = Image.open(io.BytesIO(image_bytes))
  return my_transforms(image).unsqueeze(0)
