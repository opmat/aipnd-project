import torch.cuda as cuda
import torch.optim as optim
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data as tudata
import torch.nn as nn
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as tforms
import torchvision.datasets as dsets
import torchvision.models as tmodels
from workspace_utils import active_session, keep_awake
from collections import OrderedDict
import argparse


def get_input_args():
    parser = argparse.ArgumentParser()
    # Create command line arguments using add_argument() from ArguementParser method #default = 'data/',
    parser.add_argument('data_directory', type = str, 
                    help = 'path to the data folder') 
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'Set the Model Architecture to use. Can be any of the following alexnet, vgg13, vgg16, resnet18, squeezenet1_0, densenet161') 
    parser.add_argument('--save_dir', type = str, default = '~/opt/opmat_save_dir', 
                    help = 'Set directory to save checkpoints') 
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                    help = 'Set the Learning Rate') 
    parser.add_argument('--hidden_units', type = int, default = 4096, 
                    help = 'Set the hidden unit to use') 
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'Set the Epoch to use') 
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False)
    
    return parser.parse_args()


def get_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = tforms.Compose([
        tforms.RandomResizedCrop(224),
        tforms.RandomHorizontalFlip(),
        tforms.ToTensor(),
        tforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ])
        ])
    odata_transforms = tforms.Compose([
        tforms.Resize(256),
        tforms.CenterCrop(224),
        tforms.ToTensor(),
        tforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ])
        ])
    train_datasets = dsets.ImageFolder(train_dir, train_transforms)
    valid_datasets = dsets.ImageFolder(valid_dir, odata_transforms)
    test_datasets = dsets.ImageFolder(test_dir, odata_transforms)
    
    train_loader = tudata.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = tudata.DataLoader(valid_datasets, batch_size=32) 
    test_loader = tudata.DataLoader(test_datasets, batch_size=32)
    
    return train_loader, train_datasets, train_datasets.class_to_idx, valid_loader, test_loader


def get_model(arch, nlabels, hidden_units, learning_rate):
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
    else: 
        raise ValueError('Unspected network architecture ', arch)
        
    #Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(hidden_units, nlabels),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, model.classifier, criterion, optimizer


def train_model(model, criterion, optimizer, trainloader, validloader, epochs=3, print_interval=40, device='cpu'):
#     if args.gpu and torch.cuda.is_available():
#         model.cuda()
    model.to(device=device)
    
    step = 0
    with active_session():
        
        for epoch in range(epochs):
            model.train()
            loss=0
            match = 0
            allitems = 0
            for ct, (images, labels) in enumerate(trainloader):
                model.train()
#                 if args.gpu and torch.cuda.is_available():
#                     images, labels = images.to("cuda"), labels.to("cuda")
#                 else:            
#                     images, labels = images.to("cpu"), labels.to("cpu")
                images, labels = images.to(device=device), labels.to(device=device)
                step += 1
                optimizer.zero_grad()
                outputs = model.forward(images)
                tloss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                allitems += labels.size(0)
                match += (predicted == labels).sum().item()

                tloss.backward()
                optimizer.step()
                loss += tloss.item()

                if step % print_interval == 0:
                    taccuracy = (100 * match / allitems)
                    validation = validate_model(model, criterion, validloader, device)

                    print(f"Epoch: {epoch+1}/{epochs}   ",
                      "Training Loss: {:.3f}.. ".format(loss/print_interval),
                      "Training Accuracy: {:.2f}%.. ".format(taccuracy),
                      "Valid Loss: {:.3f}.. ".format(validation['loss']),
                      "Valid Accuracy: {:.3f}".format(validation['accuracy']),
                      "Valid Accuracy: {:.3f}%".format(validation['nacc']))
                    
                    
def validate_model(model, criterion, validloader, device):
#     if args.gpu and torch.cuda.is_available():
#         model.cuda()
    model.to(device=device)
        
    model.eval()
    accuracy = 0
    loss = 0
    match = 0
    allitems = 0
    
    for ctt, (images, labels) in enumerate(validloader):
#         if args.gpu and torch.cuda.is_available():
#             images, labels = images.to("cuda"), labels.to("cuda")
#         else:            
#             images, labels = images.to("cpu"), labels.to("cpu")
        images, labels = images.to(device=device), labels.to(device=device)
        # forward pass
        with torch.no_grad():
            outputs = model.forward(images)
            # calculate loss
            vloss = criterion(outputs, labels)
            loss += vloss.item()
            ps = torch.exp(outputs)
            equality = (labels == ps.max(dim = 1)[1])
            accuracy += equality.type(torch.float64).mean().item()
            _, predicted = torch.max(outputs.data, 1)
            allitems += labels.size(0)
            match += (predicted == labels).sum().item()
                
    ret = {'loss': loss/ len(validloader),
                    'accuracy' : accuracy / len(validloader),
                    'nacc' : (100 * match / allitems)}
    return ret


def save_checkpoint(model, optimizer, save_dir, class_to_idx, classifier, arch, epochs, print_interval):
    model.class_to_idx = class_to_idx
    checkpoint = {
        'model_arch' : arch,
        'epochs' : epochs,
        'print_interval' : print_interval,
        'optimizer_state' : optimizer.state_dict(),
        'class_to_idx' : model.class_to_idx,
        'model_state': model.state_dict(),
#         'loss' : dloss,
        'classifier': classifier
    }
    
    torch.save(checkpoint, save_dir + '/my_checkpoint.ckpt')
    
    
def loadModelCheckpoint(checkpointfile='./my_checkpoint.ckpt'):

    # Load checkpoint from file
    checkpoint = torch.load(checkpointfile)
    
    vgg16model = tmodels.vgg16(pretrained=True)
    for param in vgg16model.parameters():
        param.requires_grad = False

    vgg16model.classifier = checkpoint['classifier']
    vgg16model.load_state_dict(checkpoint['model_state'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(vgg16model.classifier.parameters(), lr=0.001)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epochs']
    loss = checkpoint['loss']
    class_to_idx =  checkpoint['class_to_idx']
    
    return vgg16model, optimizer, epoch, loss, class_to_idx



def main():
    
    args = get_input_args()
    args.device = None
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        
    #Load data from provided data_directory
    train_loader, train_datasets, class_to_idx, valid_loader, test_loader = get_data(args.data_directory)
    
#     print(args)
#     print(train_datasets)
#     exit()
    
    model, classifier, criterion, optimizer = get_model(args.arch, len(train_datasets.classes), args.hidden_units, args.learning_rate)
    
    #Run Training
    train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=args.epochs, print_interval=40, device=args.device)
   
    # Save the checkpoint to save_dir
    if(args.save_dir != None):
        save_checkpoint(model, optimizer, args.save_dir, class_to_idx, classifier, args.arch, args.epochs, print_interval=40)


if __name__ == '__main__':
    main()