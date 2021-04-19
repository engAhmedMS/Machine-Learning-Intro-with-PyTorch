# Imports here
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


def arg_parse():
    parser = argparse.ArgumentParser(description="This is a parser for the train.py application")
    
    #Defining main arguments
    parser.add_argument("data_dir", type=str, help="Directory containing train and validation datasets")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="Directory to save and load trained model checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "vgg19", "densenet121", "densenet161"], help="Pretrained model for transfer learning")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=2048, help="Number of hidden layer units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Use GPU if its available")
    parser.add_argument("--batch_size", type=int, default=64, help="size of batch")
    
    return parser.parse_args()

def main():
    args = arg_parse()
    print('starting......')
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])   

    ])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    
    valid_test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225]) 
                                            ])
    
    # TODO: Load the datasets with ImageFolder
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=args.batch_size, shuffle= True)
    
    # TODO: Load the datasets with ImageFolder
    test_datasets = datasets.ImageFolder(test_dir, transform=valid_test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle= True)
    
    with open('cat_to_name.json', 'r') as f:#build 
        cat_to_name = json.load(f)
    
    #1_import pretrained model
    #"vgg16", "vgg19", "densenet121", "densenet161"
    if args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
        input_unit = 25088
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained=True) 
        input_unit = 1024
    elif args.arch == "densenet161":
        model = models.densenet161(pretrained=True)
        input_unit = 2208
    else:
        model = models.vgg16(pretrained=True)
        input_unit = 25088
        
        
    if args.device == 'cude':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    device = 'cuda'
    print('you are using: ', device)
    #2_ freez parameters (no gradient)
    for param in model.parameters():
       param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(input_unit, args.hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('fc2', nn.Linear(args.hidden_units, 102)),
                                    ('output', nn.LogSoftmax(dim=1))    
                                                    ])
                                    )
      
    
    #3_forward path
    images, labels = next(iter(train_loaders))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    epochs = args.epochs

    #3.3_start trainign
    model.to(device)
    print('training....')
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loaders:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #3.4: valdation accuracy
        else:
            valid_loss= 0
            accuracy = 0

            with torch.no_grad():
                model.eval()
                for images, labels in valid_loaders:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model.forward(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loaders)),
                  "validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loaders)),
                  "validation Accuracy: {:.3f}".format(accuracy/len(valid_loaders)))
            running_loss = 0
            model.train()


    # TODO: Do validation on the test set
    test_loss= 0
    accuracy = 0
    model.to(device)
    with torch.no_grad():
        model.eval()
        for images, labels in test_loaders:
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            valid_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("test accuracy = ", accuracy/len(test_loaders))
    model.train()

    # TODO: Save the checkpoint
    model.class_to_idx = train_datasets.class_to_idx
    check_point = {
        'model': models.vgg16(pretrained=True),
        'input_size': 25088,
        'output_size': 102,
        'learning_rate': 0.003,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer
    }

    torch.save(check_point, args.save_dir)

if __name__ == "__main__":
    main()