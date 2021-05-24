#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
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


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
# data_transforms = 
train_transforms = tforms.Compose([
    tforms.RandomResizedCrop(224),
    tforms.RandomHorizontalFlip(),
#     tforms.Resize((224,224)),
    tforms.ToTensor(),
    tforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ])
    ])
odata_transforms = tforms.Compose([
#     tforms.Resize((224,224)),
#     tforms.RandomResizedCrop(224),
    tforms.Resize(256),
    tforms.CenterCrop(224),
    tforms.ToTensor(),
    tforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ])
    ])

# TODO: Load the datasets with ImageFolder
# image_datasets = 
train_datasets = dsets.ImageFolder(train_dir, train_transforms)
valid_datasets = dsets.ImageFolder(valid_dir, odata_transforms)
test_datasets = dsets.ImageFolder(test_dir, odata_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
# dataloaders = 
train_loader = tudata.DataLoader(
    train_datasets, batch_size=64, shuffle=True) #, num_workers=2
valid_loader = tudata.DataLoader(
    valid_datasets, batch_size=32) #, shuffle=True, num_workers=2
test_loader = tudata.DataLoader(
    test_datasets, batch_size=32) #, shuffle=True, num_workers=2

print(train_datasets)
print(train_loader)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# cat_to_name


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[5]:


# TODO: Build and train your network
vgg16model = tmodels.vgg16(pretrained=True) #.densenet161(pretrained=True)
# vgg16model


# In[6]:


# for p in vgg16model.features.parameters():
#     p.require_grad = False
for param in vgg16model.parameters():
    param.requires_grad = False
    
vgg16model.parameters


# In[7]:


# train_datasets.classes
num_features0 = vgg16model.classifier[0].in_features
num_features = vgg16model.classifier[6].in_features

vgg16model.classifier = nn.Sequential(nn.Linear(num_features0, 4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(4096, 4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(num_features, len(train_datasets.classes)),
                                     nn.LogSoftmax(dim=1))
# vgg16model.classifier = nn.Sequential(OrderedDict([
#                     ('fc1', nn.Linear(25088, 4096)),
#                     ('relu1', nn.ReLU()),
#                     ('dropout1', nn.Dropout(p=0.2)),
#                     ('fc2', nn.Linear(4096, 4096)),
#                     ('relu2', nn.ReLU()),
#                     ('dropout2', nn.Dropout(p=0.3)),
#                     ('fc3', nn.Linear(4096,102)),
#                     ('output', nn.LogSoftmax(dim=1))
#                     ]))


# vgg16model


# In[8]:


criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg16model.classifier.parameters(), lr=0.001)
optimizer


# In[9]:


if torch.cuda.is_available():
    vgg16model.cuda()
    
def validate_model(model, criterion, validloader):
    if torch.cuda.is_available():
        model.cuda()
        
    model.eval()
    accuracy = 0
    loss = 0
    match = 0
    allitems = 0
    
    for ctt, (images, labels) in enumerate(validloader):
        if torch.cuda.is_available():
            images, labels = images.to("cuda"), labels.to("cuda")
        else:            
            images, labels = images.to("cpu"), labels.to("cpu")
        # forward pass
        with torch.no_grad():
            outputs = model.forward(images)
            # calculate loss
            vloss = criterion(outputs, labels)
            loss += vloss.item()
            # loss += vloss.data[0]
            ps = torch.exp(outputs)
            # ps = torch.exp(output).data
            equality = (labels == ps.max(dim = 1)[1])
            # equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type(torch.float64).mean().item()
            # accuracy += equality.type_as(torch.FloatTensor()).mean()
            _, predicted = torch.max(outputs.data, 1)
            allitems += labels.size(0)
            match += (predicted == labels).sum().item()
                
    ret = {'loss': loss/ len(validloader),
                    'accuracy' : accuracy / len(validloader),
                    'nacc' : (100 * match / allitems)}
    return ret

def train_model(model, criterion, optimizer, trainloader, validloader, epochs=2, print_interval=1):
    if torch.cuda.is_available():
        model.cuda()
        print("Cuda Used")
    
    step = 0
    with active_session():
        
        for epoch in range(epochs):
            model.train()
            loss=0
            match = 0
            allitems = 0
            for ct, (images, labels) in enumerate(trainloader):
                model.train()
                if torch.cuda.is_available():
                    images, labels = images.to("cuda"), labels.to("cuda")
                else:            
                    images, labels = images.to("cpu"), labels.to("cpu")
#                 images = Variable(images, requires_grad = False)
#                 labels = Variable(labels, requires_grad = False)
#                 with torch.no_grad():
                step += 1
                optimizer.zero_grad()
                outputs = model.forward(images)
                tloss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                allitems += labels.size(0)
                match += (predicted == labels).sum().item()

#               print(tloss)
                tloss.backward()
                optimizer.step()
                loss += tloss.item()

                if step % print_interval == 0:
                    taccuracy = (100 * match / allitems)
                    validation = validate_model(model, criterion, validloader)
#                         print(validation)

                    print(f"Epoch: {epoch+1}/{epochs}   ",
                      "Training Loss: {:.3f}.. ".format(loss/print_interval),
                      "Training Accuracy: {:.2f}%.. ".format(taccuracy),
                      "Valid Loss: {:.3f}.. ".format(validation['loss']),
                      "Valid Accuracy: {:.3f}".format(validation['accuracy']),
                      "Valid Accuracy: {:.3f}%".format(validation['nacc']))
    
# learning rate, units in the classifier, epochs
epochs = 3
print_interval = 40

# print(train_loader)
train_model(vgg16model, criterion, optimizer, train_loader, valid_loader, epochs, print_interval)


# In[10]:


print("Ready")


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[11]:


# TODO: Do validation on the test set
vgg16model.eval()
test = validate_model(vgg16model, criterion, test_loader)
print("Test Data Loss: {:.3f}-> ".format(test['loss']),
              "Test Data Accuracy: {:.3f} ~ {:.2f}%".format(test['accuracy'], (test['accuracy']*100)),
              "Test Data Accuracy(%): {:.2f}%".format(test['nacc']))
dloss = test['loss']


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[12]:


# TODO: Save the checkpoint 
vgg16model.class_to_idx = train_datasets.class_to_idx
checkpoint = {
    'model_arch' : 'vgg16',
    'epochs' : epochs,
    'print_interval' : print_interval,
    'optimizer_state' : optimizer.state_dict(),
    'class_to_idx' : vgg16model.class_to_idx,
    'model_state': vgg16model.state_dict(),
    'loss' : dloss,
    'classifier': vgg16model.classifier
}


# In[13]:


checkpoint


# In[14]:


torch.save(checkpoint, "./my_checkpoint.ckpt")


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[15]:


from PIL import Image
import numpy as np
from numpy import asarray

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #     x, y = image.size
    #     ratio = float(x)/y
    #     x_size=0
    #     y_size=0
    #     if ratio >= 1:
    #         y_size = 256
    #         x_size = int(ratio * y_size)
    #     else:
    #         x_size = 256
    #         y_size = int(x_size/ratio)

    #     image = image.resize((x_size, y_size))
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
    
    # PIL images into NumPy arrays
    np_image = asarray(image)

    # <class 'numpy.ndarray'>
#     print(type(np_image))

    #  shape
#     print(np_image.shape)
    
    #color correction
#     np_image = np.array(image)
    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))
    
    return np_image_array
    
    


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[16]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title is not None:
        ax.set_title(title)
    
    ax.imshow(image)
    
    return ax


# In[17]:


def loadMdelCheckpoint(checkpointfile='./my_checkpoint.ckpt'):
#     checkpoint = {
#     'model_arch' : 'vgg16',
#     'epochs' : epochs,
#     'print_interval' : print_interval,
#     'optimizer_state' : optimizer.state_dict(),
#     'class_to_idx' : vgg16model.class_to_idx,
#     'model_state': vgg16model.state_dict(),
#     'loss' : dloss
# }
    # Load checkpoint from file
    checkpoint = torch.load(checkpointfile)
    
    vgg16model = tmodels.vgg16(pretrained=True)
    for param in vgg16model.parameters():
        param.requires_grad = False

#     vgg16model.classifier = nn.Sequential(nn.Linear(25088, 4096),
#                                      nn.ReLU(),
#                                      nn.Dropout(0.5),
#                                      nn.Linear(4096, 4096),
#                                      nn.ReLU(),
#                                      nn.Dropout(0.5),
#                                      nn.Linear(4096, 102),
#                                      nn.LogSoftmax(dim=1))
    vgg16model.classifier = checkpoint['classifier']
    vgg16model.load_state_dict(checkpoint['model_state'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(vgg16model.classifier.parameters(), lr=0.001)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epochs']
    loss = checkpoint['loss']
    class_to_idx =  checkpoint['class_to_idx']
    
    return vgg16model, optimizer, epoch, loss, class_to_idx


# In[18]:


#test image module
import matplotlib.pyplot as plt

img_path = './flowers/test/1/image_06743.jpg'
image = Image.open(img_path)
print(image)
image = process_image(image)
imshow(torch.FloatTensor(image))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[19]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    # TODO: Implement the code to predict the class from an image file
    
    #load image
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.FloatTensor(image)
    image.unsqueeze_(0)
    if torch.cuda.is_available():
        model.to('cuda')
    
    if torch.cuda.is_available():
        images= images.to("cuda")
    else:            
        images = images.to("cpu")
    
    with torch.no_grad():
        output = model.forward(image)
        results = torch.exp(output).topk(topk)
    
    probs = results[0][0]
    classes = results[1][0]
    
    return probs, classes
    


# In[20]:


#load model checkpoint
mymodel, myoptimizer, myepoch, myloss, myclass_to_idx = loadMdelCheckpoint()
# img_path = test_dir + '/1/image_06760.jpg'
img_path = './flowers/test/1/image_06743.jpg'
probabilities, classes = predict(img_path, mymodel)
print(probabilities)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[21]:


def check_sanity(img_path):
    # TODO: Display an image along with the top 5 classes
    #load model checkpoint
    mymodel, myoptimizer, myepoch, myloss, myclass_to_idx = loadMdelCheckpoint()
    
    probabilities, classes = predict(img_path, mymodel)
    # print(type(probabilities))
    # print(probabilities.cpu().detach().numpy())
    # print(classes.cpu().detach().numpy())

    prob = probabilities.cpu().detach().numpy()
    classe = classes.cpu().detach().numpy()

    labels = [cat_to_name[str(x)] for x in classe]
    # print(labels)

    image = Image.open(img_path)
    image = process_image(image)
    imshow(torch.FloatTensor(image), title=labels[0])

    #Plot Chart
    fig,axs = plt.subplots(sharey=True)
    order = np.arange(len(labels))[::-1]

    plt.yticks(order, labels)
    axs.barh(order, prob)

    plt.show()
    
#Run Check
img_path = test_dir + '/1/image_06760.jpg'
# img_path = './flowers/test/1/image_06743.jpg'
check_sanity(img_path)

