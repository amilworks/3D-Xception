#!/usr/bin/python
from model import Xception3d
from dataloader import DeepFakeDataset
import torch
import torch.optim as optim
import torch.nn as nn
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
torch.backends.cudnn.benchmark = True




#----------------------------------------------------------------------------
# Dataloader
print('='*70)
print("Loading Data...")
print('='*70)
trainset = DeepFakeDataset('data/dfdc_train_part_', num_frames=40, num_folders=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,shuffle=True, num_workers=4, pin_memory=True)
print("---> Successfully Loaded Data!\n")

#----------------------------------------------------------------------------
# He Initialization, can be modified to whatever you want

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


#----------------------------------------------------------------------------
# Load and Optimize model using Apex, define SGD parameters, distribute across GPUs
print('='*70)
print("Loading Model...")
print('='*70)
learning_rate=1e-05
momentm=0.9
wt_decay=1e-05

print(
    """\nModel Parameters
    
Learning Rate        Momentum         Weight Decay     
==============     ==============    ==============
    {}                {}              {}
    """.format(learning_rate, momentm,wt_decay)
)

device = torch.device( "cuda")
criterion = nn.BCEWithLogitsLoss()
model = Xception3d(num_classes=1).apply(init_weights).to(device)
print("Optimizing Model with Apex...")
optimizer = apex.optimizers.FusedSGD(model.parameters(), lr=learning_rate,momentum=momentm, weight_decay=wt_decay)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = apex.parallel.convert_syncbn_model(model)
model = nn.DataParallel(model)

lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
print("\n---> Successfully Loaded and Optimized Model...\n")

#----------------------------------------------------------------------------
# Training Loop 
print('='*70)
print("Training...")
print('='*70)
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(trainloader):

        vid   = data['video'].to(device)
#         aud   = data['audio'].to(device)
#         print('Audio',i, aud.shape)
        label = data['label'].to(device)
#         print(label)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(vid)
        loss = criterion(outputs, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        lr_sched.step()
        # print statistics
        running_loss += loss.item()
        print(f'Label: {label} \t', f'Loss: {loss}')
        if i % 6 == 5:    # print every 5 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            print("Learning Rate {}:".format(epoch), optimizer.param_groups[0]['lr'])
            running_loss = 0.0

print('Finished Training')