from PIL import Image
from torchvision import transforms

def Bicubic_for_deployment(lr_image):
    w, h = lr_image.size
    sr_image = transforms.functional.resize(lr_image, size=(h*4, w*4), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    return sr_image