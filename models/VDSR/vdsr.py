import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import os
from math import sqrt
import torch.nn.functional as F

#define class Block contain conv and relu layer
class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=18):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Block, num_blocks)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                          
    def make_layer(self, block, num_layers):
        layers=[]
        for _ in range(num_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(residual, out)
        return out
    
    def inference(self, x):
        """
        x is a PIL image
        """
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        with torch.no_grad():
            x = ToTensor()(x).unsqueeze(0).to(device)
            x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
            x = self.forward(x).clamp(0, 1)
            x = x.cpu()  # Move tensor back to CPU for conversion to PIL image
            x = Image.fromarray((x.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))
        
        return x

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    # model = torch.load(current_dir + '/vdsr_checkpoint.pth', map_location=device)
    model = VDSR()
    checkpoint = torch.load(current_dir + '/vdsr_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        input_image = Image.open('images/demo.png')
        output_image = model.inference(input_image) 
    print(input_image.size, output_image.size)
