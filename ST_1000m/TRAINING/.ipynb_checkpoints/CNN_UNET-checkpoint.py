import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from torch import optim
import progressbar
from CNN_tools import *
from variables import nb_dx,alpha1,alpha2
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#CNN using 4 vertical levels
class CNN_UNET_SURF(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(    
        nn.Conv2d(28, 64, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(64)
             )
        
        self.conv2 = nn.Sequential(    
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(128)
             )
        
        self.conv3 = nn.Sequential(    
        nn.Conv2d(128, 256, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(256)
             )
        
        self.conv_up1 = nn.Sequential(    
        nn.Conv2d(128, 64, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 8, kernel_size = 3, padding = 1, bias=False)
             )
        
        self.conv_up2 = nn.Sequential(    
        nn.Conv2d(256, 128, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(128)
             )
        
        self.convTrans3 = nn.ConvTranspose2d(256,128,kernel_size = 2,padding = 0,stride=2)
        self.convTrans2 = nn.ConvTranspose2d(128,64,kernel_size = 2,padding = 0,stride=2)
        
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=2, end_dim=- 1)
        self.maxpool2d = nn.MaxPool2d(2)
        self.avgpool2d = nn.AvgPool2d(2)
        self.relu = nn.ReLU(True)
        
        
    def forward(self, z, y_filter,y):
        
        y_pred = torch.zeros(y.shape).to('cuda')
        y_pred[:,:,49:51,49:51] = 0.25 # Initialisation of y_pred
        
        y1 = torch.cat((z,y_pred),dim=1) 
        y1 = self.conv1(y1)
        y2 = self.maxpool2d(y1)
        y2 = self.conv2(y2)
        y3 = self.maxpool2d(y2)
        y3 = self.conv3(y3)
        
        y3 = self.convTrans3(y3)
        y3 = torch.cat((y2,y3),dim=1)
        y2 = self.conv_up2(y3)
        
        y2 = self.convTrans2(y2)
        y2 = torch.cat((y1,y2),dim=1)
        y1 = self.conv_up1(y2)
        
        y_hat = self.flatten(y1)
        y_hat = self.relu(y_hat)
        y_hat = self.softmax(torch.log(y_hat+1e-10)) 
        y_hat = y_hat.view(y_hat.shape[0],8,nb_dx,nb_dx)
        
        return y_hat 
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = optim.Adam(self.parameters(),lr= lr, betas=(0.5, 0.999),weight_decay=0)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        loss = 0

        
        for i in range(0,8):
            loss = loss + alpha1*Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:]) + alpha2*Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss = loss / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_train", loss, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_train", loss, on_epoch=True, on_step = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        
        loss_filter = 0
        loss_no_filter = 0

        for i in range(0,8):
            loss_filter = loss_filter + Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:])
            loss_no_filter = loss_no_filter + Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss_filter = loss_filter / 8
        loss_no_filter = loss_no_filter / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_filter_validation", loss_filter, on_epoch=True, on_step = True)
        self.log("loss_no_filter_validation", loss_no_filter, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_validation", loss_filter_200m, on_epoch=True, on_step = True)
        
        return loss_no_filter


#CNN using 4 vertical levels
class CNN_UNET_4L(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(    
        nn.Conv2d(76, 64, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(64)
             )
        
        self.conv2 = nn.Sequential(    
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(128)
             )
        
        self.conv3 = nn.Sequential(    
        nn.Conv2d(128, 256, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(256)
             )
        
        self.conv_up1 = nn.Sequential(    
        nn.Conv2d(128, 64, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 8, kernel_size = 3, padding = 1, bias=False)
             )
        
        self.conv_up2 = nn.Sequential(    
        nn.Conv2d(256, 128, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(128)
             )
        
        self.convTrans3 = nn.ConvTranspose2d(256,128,kernel_size = 2,padding = 0,stride=2)
        self.convTrans2 = nn.ConvTranspose2d(128,64,kernel_size = 2,padding = 0,stride=2)
        
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=2, end_dim=- 1)
        self.maxpool2d = nn.MaxPool2d(2)
        self.avgpool2d = nn.AvgPool2d(2)
        self.relu = nn.ReLU(True)
        
        
    def forward(self, z, y_filter,y):
        
        y_pred = torch.zeros(y.shape).to('cuda')
        y_pred[:,:,49:51,49:51] = 0.25 # Initialisation of y_pred

        y1 = torch.cat((z,y_pred),dim=1) 
        y1 = self.conv1(y1)
        y2 = self.maxpool2d(y1)
        y2 = self.conv2(y2)
        y3 = self.maxpool2d(y2)
        y3 = self.conv3(y3)
        
        y3 = self.convTrans3(y3)
        y3 = torch.cat((y2,y3),dim=1)
        y2 = self.conv_up2(y3)
        
        y2 = self.convTrans2(y2)
        y2 = torch.cat((y1,y2),dim=1)
        y1 = self.conv_up1(y2)
        
        y_hat = self.flatten(y1)
        y_hat = self.relu(y_hat)
        y_hat = self.softmax(torch.log(y_hat+1e-10)) 
        y_hat = y_hat.view(y_hat.shape[0],8,nb_dx,nb_dx)
        
        return y_hat 
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = optim.Adam(self.parameters(),lr= lr, betas=(0.5, 0.999),weight_decay=0)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        loss = 0

        
        for i in range(0,8):
            loss = loss + alpha1*Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:]) + alpha2*Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss = loss / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_train", loss, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_train", loss, on_epoch=True, on_step = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        
        loss_filter = 0
        loss_no_filter = 0

        for i in range(0,8):
            loss_filter = loss_filter + Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:])
            loss_no_filter = loss_no_filter + Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss_filter = loss_filter / 8
        loss_no_filter = loss_no_filter / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_filter_validation", loss_filter, on_epoch=True, on_step = True)
        self.log("loss_no_filter_validation", loss_no_filter, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_validation", loss_filter_200m, on_epoch=True, on_step = True)
        
        return loss_no_filter

    #CNN using 4 vertical levels
class CNN_UNET_generic(pl.LightningModule):
    def __init__(self,kernel_size,padding,bias,p_dropout,nlayer0,nb_inputs):
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.p_dropout = p_dropout
        self.nlayer0 = nlayer0
        self.nb_inputs = nb_inputs
        #76 4L or 28 surface
        super().__init__()
        
        self.conv1 = nn.Sequential(    
        nn.Conv2d(self.nb_inputs, nlayer0, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.ReLU(True),
        nn.BatchNorm2d(nlayer0),
        nn.Conv2d(nlayer0, nlayer0, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.BatchNorm2d(nlayer0)
             )
        
        self.conv2 = nn.Sequential(    
        nn.Conv2d(nlayer0, nlayer0*2, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.ReLU(True),
        nn.BatchNorm2d(nlayer0*2),
        nn.Conv2d(nlayer0*2, nlayer0*2, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.BatchNorm2d(nlayer0*2)
             )
        
        self.conv3 = nn.Sequential(    
        nn.Conv2d(nlayer0*2, nlayer0*4, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.ReLU(True),
        nn.BatchNorm2d(nlayer0*4),
        nn.Conv2d(nlayer0*4, nlayer0*4, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.BatchNorm2d(nlayer0*4)
             )
        
        self.conv_up1 = nn.Sequential(    
        nn.Conv2d(nlayer0*2, nlayer0, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.ReLU(True),
        nn.BatchNorm2d(nlayer0),
        nn.Conv2d(nlayer0, nlayer0, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.BatchNorm2d(nlayer0),
        nn.Conv2d(nlayer0, 8, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias)
             )
        
        self.conv_up2 = nn.Sequential(    
        nn.Conv2d(nlayer0*4, nlayer0*2, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.ReLU(True),
        nn.BatchNorm2d(nlayer0*2),
        nn.Conv2d(nlayer0*2, nlayer0*2, kernel_size = self.kernel_size, padding = self.padding, bias=self.bias),
        nn.BatchNorm2d(nlayer0*2)
             )
        
        self.convTrans3 = nn.ConvTranspose2d(nlayer0*4,nlayer0*2,kernel_size = 2,padding = 0,stride=2)
        self.convTrans2 = nn.ConvTranspose2d(nlayer0*2,nlayer0,kernel_size = 2,padding = 0,stride=2)
        
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=2, end_dim=- 1)
        self.maxpool2d = nn.MaxPool2d(2)
        self.avgpool2d = nn.AvgPool2d(2)
        self.relu = nn.ReLU(True)
        
        self.dropout = nn.Dropout(p_dropout)
        
        
    def forward(self, z, y_filter,y):
        
        y_pred = torch.zeros(y.shape).to('cuda')
        y_pred[:,:,49:51,49:51] = 0.25 # Initialisation of y_pred

        y1 = torch.cat((z,y_pred),dim=1) 
        y1 = self.conv1(y1)
        y2 = self.maxpool2d(y1)
        y2 = self.conv2(y2)
        y3 = self.maxpool2d(y2)
        y3 = self.conv3(y3)
        
        y3 = self.convTrans3(y3)
        y3 = torch.cat((y2,y3),dim=1)
        y2 = self.conv_up2(y3)
        
        y2 = self.convTrans2(y2)
        y2 = torch.cat((y1,y2),dim=1)
        y1 = self.conv_up1(y2)
        
        y_hat = self.flatten(y1)
        y_hat = self.relu(y_hat)
        y_hat = self.softmax(torch.log(y_hat+1e-10)) 
        y_hat = y_hat.view(y_hat.shape[0],8,nb_dx,nb_dx)
        
        return y_hat 
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = optim.Adam(self.parameters(),lr= lr, betas=(0.5, 0.999),weight_decay=0)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        loss = 0

        
        for i in range(0,8):
            loss = loss + alpha1*Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:]) + alpha2*Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss = loss / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_train", loss, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_train", loss, on_epoch=True, on_step = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        
        loss_filter = 0
        loss_no_filter = 0

        for i in range(0,8):
            loss_filter = loss_filter + Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:])
            loss_no_filter = loss_no_filter + Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss_filter = loss_filter / 8
        loss_no_filter = loss_no_filter / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_filter_validation", loss_filter, on_epoch=True, on_step = True)
        self.log("loss_no_filter_validation", loss_no_filter, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_validation", loss_filter_200m, on_epoch=True, on_step = True)
        
        return loss_no_filter


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.FloatTensor(sample)



class cnn_parallele(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            
            nn.Conv2d(52, 52, kernel_size = 5, padding = 2),
            nn.ReLU(True),    
            nn.MaxPool2d(4),
            nn.Flatten(start_dim=1, end_dim=- 1),       
            nn.Linear(20*20*52,10),
            nn.Linear(10,4000)      
            #nn.Dropout(0.5)
         )
        
        self.decoder = nn.Sequential(    
            
            nn.ConvTranspose2d(10,1,kernel_size = 6,padding = 1,stride=4),
            nn.Flatten(start_dim=1, end_dim=- 1),
            #nn.Softmax(dim=1)
                     
        )
        
        self.cnn2 = nn.Sequential(    

            nn.Conv2d(52, 26, kernel_size = 5, padding = 2),
            nn.ReLU(True),
            nn.Conv2d(26, 1, kernel_size = 5, padding = 2),
            nn.Flatten(start_dim=1, end_dim=- 1),
            
            #nn.Softmax(dim=1)

    )
        self.softmax = nn.Softmax(dim=1)
        self.relu=nn.ReLU(True)
        
    def forward(self, z):
        
        # Decoder - Encoder CNN
        out1 = self.encoder(z)
        out1 = out1.view(out1.shape[0],10,20,20)
        out1 = self.decoder(out1)
        # First CNN
        out2 = self.cnn2(z)
        out2 = self.relu(out2)
        out2 = self.softmax(torch.log(out2))
        #We merge outputs
        out = torch.mul(out1,out2)

        out = self.relu(out)
        out = self.softmax(torch.log(out))
        out = out.view(out.shape[0],1,100,100)
        
        return out
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = optim.Adam(self.parameters(),lr= lr, betas=(0.5, 0.999),weight_decay=0)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #loss = bhatta_loss(y_hat, y)
        loss = Bhatta_loss(y_hat, y)
        self.log("loss_filter_200m_train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = Bhatta_loss(y_hat, y)
        self.log('loss_filter_200m_validation', val_loss, on_step=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = Bhatta_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss
    

#CNN using 4 vertical levels
class CNN_UNET_SURF_1step(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(    
        nn.Conv2d(28, 64, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(128)
             )
        
        self.conv_up1 = nn.Sequential(    
        nn.Conv2d(128, 64, kernel_size = 3, padding = 1, bias=False),
        nn.ReLU(True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=False),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 8, kernel_size = 3, padding = 1, bias=False)
             )
        
        self.softmax = nn.Softmax(dim=2)
        self.flatten = nn.Flatten(start_dim=2, end_dim=- 1)
        self.maxpool2d = nn.MaxPool2d(2)
        self.avgpool2d = nn.AvgPool2d(2)
        self.relu = nn.ReLU(True)
        
        
    def forward(self, z, y_filter,y):
        
        y_pred = torch.zeros(y.shape).to('cuda')
        y_pred[:,:,49:51,49:51] = 0.25 # Initialisation of y_pred
        
        y1 = torch.cat((z,y_pred),dim=1) 
        y1 = self.conv1(y1)

        y1 = self.conv_up1(y1)
        
        y_hat = self.flatten(y1)
        y_hat = self.relu(y_hat)
        y_hat = self.softmax(torch.log(y_hat+1e-10)) 
        y_hat = y_hat.view(y_hat.shape[0],8,nb_dx,nb_dx)
        
        return y_hat 
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = optim.Adam(self.parameters(),lr= lr, betas=(0.5, 0.999),weight_decay=0)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        loss = 0

        
        for i in range(0,8):
            loss = loss + alpha1*Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:]) + alpha2*Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss = loss / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_train", loss, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_train", loss, on_epoch=True, on_step = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        z, y_f, y = batch
        y_hat = self(z,y_f,y)
        
        loss_filter = 0
        loss_no_filter = 0

        for i in range(0,8):
            loss_filter = loss_filter + Bhatta_loss(y_hat[:,i,:,:], y_f[:,i,:,:])
            loss_no_filter = loss_no_filter + Bhatta_loss(y_hat[:,i,:,:], y[:,i,:,:])
            
        loss_filter = loss_filter / 8
        loss_no_filter = loss_no_filter / 8
        
        loss_filter_200m = Bhatta_loss(y_hat[:,-1,:,:], y_f[:,-1,:,:])
        
        self.log("loss_filter_validation", loss_filter, on_epoch=True, on_step = True)
        self.log("loss_no_filter_validation", loss_no_filter, on_epoch=True, on_step = True)
        self.log("loss_filter_200m_validation", loss_filter_200m, on_epoch=True, on_step = True)
        
        return loss_no_filter
