# %%
# Imports 
import os  
import pandas as pd  
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

# %%
transformPipeline = transforms.Compose([
    transforms.Resize((224,224)),    
    transforms.ToTensor(),
])

# %%
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class celebDataset(Dataset):
    
    def __init__(self, csvFile: str, imageDirectory: str, bboxFile: str, transform=None) -> None:
        """
        Args:
            csvFile (str): Path to the CSV file containing image labels.
            imageDirectory (str): Path to the image directory.
            bboxFile (str): Path to the CSV file containing bounding box data.
            transform: Transformations to apply to images (optional).
        """
        self.imageDirectory = imageDirectory
        self.imageName = pd.read_csv(csvFile)["image_id"]
        self.imageLabels = pd.read_csv(csvFile)["Arched_Eyebrows"]
        self.transform = transform
        
        # Load bounding box data
        self.bboxData = pd.read_csv(bboxFile)

    def __len__(self) -> int:
        return len(self.imageLabels)
    
    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        # Get Image
        image_name = self.imageName.iloc[index]
        image_path = os.path.join(self.imageDirectory, image_name)
        image = Image.open(image_path)

        # Get Label
        label = int(self.imageLabels.iloc[index])
        
        # Change label from -1 to 0
        if label == -1:
            label = 0
        
        x1 = 40
        y1 = 74
        width = 100
        height = 100
        # Crop the image using the bounding box (left, upper, right, lower)
        # image = image.crop((x1, y1, x1 + height, y1 + width))

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label  # Return the image tensor and label


# %%
image_dir = "../img_align_celeba"
csv = "../Arched_Eyebrows.csv"
bbox = "../face_image_bbox.csv"
dataset = celebDataset(csv, image_dir,bbox, transformPipeline)

# %%
print(len(dataset))

# %%
# plot few images 
fig, ax = plt.subplots(1, 5, figsize=(20, 5))
for i in range(5):
    img, label = dataset[i]
    ax[i].imshow(img.permute(1, 2, 0).numpy())
    ax[i].set_title(f"Label: {label}")
    ax[i].axis("off")
plt.show()


# %%


# %%
image,label = dataset[2]
print(image.shape)

# %%
# plot this image
plt.imshow(image.permute(1,2,0))
plt.title(label)
plt.show()

# %%
trainSize = int(0.8*len(dataset))
validationSize = int(0.2*(len(dataset)))

    
# These are remaining data points
offset = (len(dataset)-trainSize-validationSize)

print(f"Training Set Size: {trainSize}, Validation Set Size: {validationSize+offset}")

# %%
# Split the dataset
trainSet, validSet = torch.utils.data.random_split(dataset,[trainSize,validationSize+offset])

# %%
# make the dataloader
batchsize = 128

trainLoader = DataLoader(trainSet,batch_size=batchsize,shuffle=True)
validLoader = DataLoader(validSet,batch_size=batchsize,shuffle=True)


# %%

import torchvision.models as models
import torch.nn as nn

# Load pre trained resnet model
model = models.resnet18(weights="ResNet18_Weights.DEFAULT")

numFeatures = model.fc.in_features
print(numFeatures)
model.fc = nn.Sequential(
    nn.Linear(numFeatures, 256),  # First hidden layer
    nn.ReLU(),                    # Activation function
    nn.Dropout(0.8),               # Dropout for regularization (optional)
    nn.Linear(256, 2),           # Output layer for binary classification
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
from torchinfo import summary

# Assuming 'model' is your defined ResNet18 model
summary(model, input_size=(1, 3, 224, 224))  # Note: Batch size of 1 is specified here


# %%
# Loss Function
import torch.nn as nn 
lossFunction = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)


# %%
print(device)

# %%
import torch
from tqdm import tqdm
from colorama import Fore, Style

# Training Loop with Progress Bar and Colorful Output
numEpochs = 2
model.to(device)

for epoch in range(numEpochs):
    model.train()
    running_loss = 0.0
    print(f"{Fore.CYAN}Epoch [{epoch + 1}/{numEpochs}]{Style.RESET_ALL}")
    
    # Use tqdm for progress bar in training
    train_loader_tqdm = tqdm(enumerate(trainLoader), total=len(trainLoader), desc=f"{Fore.YELLOW}Training Epoch {epoch + 1}{Style.RESET_ALL}")
    
    for batchIndex, (images, labels) in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # if batchIndex % 100 == 0:
        train_loader_tqdm.set_postfix(loss=loss.item())
            
    # Print epoch loss
    avg_loss = running_loss / len(trainLoader)
    print(f"{Fore.GREEN}Epoch {epoch + 1}, Average Training Loss: {avg_loss:.4f}{Style.RESET_ALL}")
    
    # Validation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(validLoader, desc=f"{Fore.YELLOW}Validating{Style.RESET_ALL}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    accuracy = correct / total
    print(f"{Fore.BLUE}Epoch {epoch + 1}, Validation Accuracy: {accuracy:.4f}{Style.RESET_ALL}")

    # Save the model pth file
torch.save(model.state_dict(), "model_withoutCrop.pth")



# %%


# %%
testSet = celebDataset("../testSet.csv", image_dir,bbox, transformPipeline)

# %%
print(len(testSet)) 

# %%
testLoader = DataLoader(testSet, batch_size=batchsize, shuffle=False)

# %%
# Visualize few images on test set
fig, ax = plt.subplots(1, 5, figsize=(20, 5))
for i in range(5):
    img, label = testSet[i]
    ax[i].imshow(img.permute(1, 2, 0).numpy())
    ax[i].set_title(f"Label: {label}")
    ax[i].axis("off")
plt.show()


# %%
# move to device
model.to(device)


# %%
all_labels = []
all_preds = []
all_probs = []

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(testLoader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probability for class 1
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_accuracy = correct / total
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Compute precision, recall, F1 score
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix:\n{conf_matrix}")

# Beautify confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.show()

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve with improved visualization
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%


# %%
# Test on a single image
from PIL import Image
import torch
from torchvision import transforms

# Load the image
image_path = "../TestImages/test5.png"
image = Image.open(image_path)

# Apply transformations
transformPipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the image and preprocess it
def preprocess_image(image_path):
    image = Image.open(image_path)

    # If the image has 4 channels (RGBA), convert it to 3 channels (RGB)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply transformations (resize, normalize, etc.)
    image_tensor = transformPipeline(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

image_tensor = preprocess_image(image_path)

# plot the image
plt.imshow(image)

# Load the model
model = models.resnet18()
numFeatures = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(numFeatures, 256),  # First hidden layer
    nn.ReLU(),                    # Activation function
    nn.Dropout(0.8),               # Dropout for regularization (optional)
    nn.Linear(256, 2),           # Output layer for binary classification
)

model.load_state_dict(torch.load("model_wcrop.pth"))
model.eval()

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    _, predicted = torch.max(output, 1)
    
    print(f"Predicted Class: {predicted.item()}")
    print(f"Probability of Class 1: {probs[0, 1].item()}")
    
    if predicted.item() == 0:
        print("Prediction: No Arched Eyebrows")
    else:
        print("Prediction: Arched Eyebrows")
        
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    

# %%
all_labels = []
all_preds = []
all_probs = []

model.eval()
model.to(device)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(testLoader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probability for class 1
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_accuracy = correct / total
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Compute precision, recall, F1 score
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print(f"Confusion Matrix:\n{conf_matrix}")

# Beautify confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.show()

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve with improved visualization
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%



