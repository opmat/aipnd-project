import torch 
import argparse
from PIL import Image
import torchvision
import json
# from torchvision import datasets, transforms, models
import torchvision.transforms as tforms
import torchvision.datasets as dsets
import torchvision.models as tmodels
from workspace_utils import active_session, keep_awake
import numpy as np
from numpy import asarray
import torch.nn as nn
import torch.optim as optim

def get_input_args():
    parser = argparse.ArgumentParser()
    # Create command line arguments using add_argument() from ArguementParser method #default = 'data/',
    parser.add_argument('image_path', type = str, 
                    help = 'path to the image to predict') 
    parser.add_argument('checkpoint', type = str, 
                    help = 'the checkpoint') 
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'Set the top K value') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'Set the JSON file containing the real name') 
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False)
    
    return parser.parse_args()


def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(img_path)
    image.thumbnail((256,256))
    #center crop to 224x224
    width, height = image.size
    new_width, new_height = 224, 224
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
#     imageloader =  tforms.Compose([
#         tforms.Resize(256),
#         tforms.CenterCrop(224),
#         tforms.ToTensor(),
#         tforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                              std = [ 0.229, 0.224, 0.225 ])
#         ])
    
#     image = imageloader(image)
    
    # PIL images into NumPy arrays
    np_image = asarray(image)

    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))
    
    return np_image_array, torch.FloatTensor(np_image_array)


def predict(image_path, model, topk=3, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()    
    #load image
#     image = Image.open(image_path)
    image_array, image = process_image(image_path)
#     image = torch.FloatTensor(image)
    image.unsqueeze_(0)
    model.to(device=device)
#     if args.gpu and torch.cuda.is_available():
#         model.to('cuda')
    
#     if args.gpu and torch.cuda.is_available():
#         image= image.to("cuda")
#     else:            
#         image = image.to("cpu")
    image = image.to(device=device)
    with torch.no_grad():
        output = model.forward(image)
        results = torch.exp(output).topk(topk)
    
    probabilities = results[0][0]
    classes = results[1][0]
    
    prob = probabilities.cpu().detach().numpy()
    classe = classes.cpu().detach().numpy()
    
    return prob, classe
    

def get_real_name(topk_classes, cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[str(x)] for x in topk_classes]
    return labels


def loadModelCheckpoint(checkpointfile='./my_checkpoint.ckpt'):

    # Load checkpoint from file
    checkpoint = torch.load(checkpointfile)
    
    arch = checkpoint['model_arch']
    
    if arch == 'vgg16':
        model = tmodels.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = tmodels.vgg13(pretrained=True)
    elif arch == 'alexnet':
        model = tmodels.alexnet(pretrained=True)
    elif arch == 'resnet18':
        model = tmodels.resnet18(pretrained=True)
    elif arch == 'squeezenet1_0':
        model = tmodels.squeezenet1_0(pretrained=True)
    elif arch == 'densenet161':
        model = tmodels.densenet161(pretrained=True)
    
#     model = tmodels.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epochs']
    loss = 0.70
    class_to_idx =  checkpoint['class_to_idx']
    
    return model, optimizer, epoch, loss, class_to_idx



def main():
    
    args = get_input_args()
    args.device = None
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
      
    #Load Checkpoint
    model, optimizer, epoch, loss, class_to_idx = loadModelCheckpoint(args.checkpoint)
    
    #Predict
    prob, classe = predict(args.image_path, model, args.top_k, device=args.device)
    
    #get Real Names
    labels = get_real_name(classe, args.category_names)
    
    print(f"The Top {args.top_k} Predicted Names of the Flower with their probabilities are :")
    for i in range(args.top_k):
        print(f"{i+1}. {labels[i]} with probability of {prob[i]*100:.2f}%")


if __name__ == '__main__':
    main()