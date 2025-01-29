import os 
import torch
import torch.optim.adam 
import data_setup , engine , model_builder , utils
from torchvision import transforms

NUM_EPOCHS = 140
BATCH_SIZE = 4
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = "/home/user/Desktop/vit/going_modular/data/pizza_steak_sushi/train"
test_dir = "/home/user/Desktop/vit/going_modular/data/pizza_steak_sushi/test"
device = "cuda" if torch.cuda.is_available() else "cpu"
train_transform = transforms.Compose([
 transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 0.7)),

transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomVerticalFlip(p=0.5),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
transforms.Resize((64,64)),

transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataLoader , test_dataLoader , class_names = data_setup.create_dataloaders(test_dir=test_dir,train_dir=train_dir,batch_size=BATCH_SIZE,train_transform=train_transform , test_transform=test_transform)
model=model_builder.TinyVGG(hidden_units=HIDDEN_UNITS,input_shape=3,output_shape=len(class_names)).to(device)
loss_fn =torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters() , lr=LEARNING_RATE , weight_decay=1e-4)
engine.train(model=model , optimizer=optimizer , loss_fn=loss_fn , test_dataloader=test_dataLoader , train_dataloader= train_dataLoader , epochs=NUM_EPOCHS , device=device)
utils.save_model(model=model , model_name="firsttry.pth" , target_dir="models")