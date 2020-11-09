import torch
import torch.nn as nn 


class CNN_4Conv(nn.Module):
    def __init__(self,batchNorm,outKernels,linDim):
        super(CNN_4Conv,self).__init__()
        self.outKernels = outKernels
        self.batchNorm = batchNorm
        self.linDim = linDim
       
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.outKernels,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.outKernels,out_channels=self.outKernels,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.outKernels,out_channels=self.outKernels,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.outKernels,out_channels=self.outKernels,kernel_size=3,stride=1)

        if (self.batchNorm):
            self.conv1BN = nn.BatchNorm2d(self.outKernels)       
            self.conv2BN = nn.BatchNorm2d(self.outKernels)       
            self.conv3BN = nn.BatchNorm2d(self.outKernels)
            self.conv4BN = nn.BatchNorm2d(self.outKernels)
            self.lin1BN = nn.BatchNorm1d(self.linDim)
            self.lin2BN = nn.BatchNorm1d(16)

        self.pool = nn.MaxPool2d(2,2)
        # need to pad 1 pixel b/c dims become odd later
        self.poolE = nn.MaxPool2d(2,2,padding=1)
        self.relu = nn.ReLU()
        # for 4 conv layers
        self.fc1 = nn.Linear(2*2*self.outKernels,self.linDim)
        self.fc2 = nn.Linear(self.linDim,16)
        self.fc3 = nn.Linear(16,10)



    def forward(self,x):
        if self.batchNorm:
            x = self.pool(self.relu(self.conv1BN(self.conv1(x))))
            x = self.poolE(self.relu(self.conv2BN(self.conv2(x))))
            x = self.poolE(self.relu(self.conv3BN(self.conv3(x))))
            x = self.pool(self.relu(self.conv4BN(self.conv4(x))))
            x = x.view(-1,2*2*self.outKernels) 
            x = self.lin1BN(self.relu(self.fc1(x)))
            x = self.lin2BN(self.relu(self.fc2(x)))
            x = (self.fc3(x))
            return x

        else:
            x = self.pool(self.relu(self.conv1(x)))
            x = self.poolE(self.relu(self.conv2(x)))
            x = self.poolE(self.relu(self.conv3(x)))
            x = self.pool(self.relu(self.conv4(x)))
            x = x.view(-1,2*2*self.outKernels) 
            x = (self.relu(self.fc1(x)))
            x = (self.relu(self.fc2(x)))
            x = (self.fc3(x))
            return x
        
        


from torchsummary import summary

# net = CNN_4Conv(batchNorm=False,outKernels=10,linDim=32)

# print(
#     summary(net,(3,56,56))
# )