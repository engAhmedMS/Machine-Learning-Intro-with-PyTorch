import numpy as np
import torch
from torch import optim
from torch import nn
from torchvision import transforms, datasets, models
from collections import OrderedDict
from PIL import Image
import json
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/58/image_02663.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    
    for param in model.parameters():
        param.requires_grad = False 
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    x = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])   

                                    ])
        
    image = transform(x)
    return image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #changing model do evaluation mode
    model.eval()
    
    #Image processing
    image = process_image(image_path)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    #Loading Model   
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(topk, dim=1)
        top_ps = top_ps.detach().cpu().numpy().tolist()[0]
        top_classes = top_classes.detach().cpu().numpy().tolist()[0]
    return top_ps, top_classes




def main():
	args = arg_parse()
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
	image_path = args.image_path
	model = load_checkpoint(args.checkpoint)
    predict(image_path, model, args.top_k)


if __name__ == "__main__":
    main()