import os
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor, functional as TF

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )
    
    def forward(self, x):
        return x + self.conv_block(x)
    
class GeneratorResnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResnet, self).__init__()
        #first layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())
        
        #Residual blocks
        res_blocks=[]
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        #second conv layer after res blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))
        upsampling=[]
        for _ in range(2):
            upsampling+=[
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

        
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out.clamp(0, 1)
    
    def inference(self, x):
        """
        x is a PIL image
        """
        self.eval()
        with torch.no_grad():
            x = ToTensor()(x).unsqueeze(0)
            x = self.forward(x.to(self.device ))
            x = Image.fromarray((x.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))
        return x
    
    def test(self, x):
        """
        x is a tensor
        """
        self.eval()
        with torch.no_grad():
            x = self.forward(x.to(self.device))
        return x
    
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))

    model = GeneratorResnet()
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model = model.load_state_dict(torch.load(current_dir + '/srgan_checkpoint.pth', map_location=device)).to(device)
    model.eval()
    with torch.no_grad():
        input_image = Image.open('images/demo.png')
        input_image = ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(device)
        output_image = model.test(input_image)
        print(output_image.max())