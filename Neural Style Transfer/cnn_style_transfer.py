import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size  # Get original size

    if shape is not None:
        size = shape if isinstance(shape, (tuple, list)) else (shape, shape)
    else:
        # Scale the longest side to max_size while maintaining aspect ratio
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            size = (int(height * scale),int(width * scale))  # Resize while keeping aspect ratio
        else:
            size = (height, width)  # Keep original size if it's already smaller than max_size

    transform = transforms.Compose([
        transforms.Resize(size),  # Resize while keeping aspect ratio
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension # (C, H, W) -> (1, C, H, W)
    return image


# Function to deprocess (unnormalize & convert to NumPy)
def deprocess_image(image):
    image = image.detach().cpu()

    # Unnormalize the image
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # image = image * std + mean

    image = image.clamp(0, 1)  # Ensure values are in [0,1]
    image = image.squeeze(0).numpy().transpose(1, 2, 0)  # Shape: (H, W, C)

    return image


# Content Loss Class
class ContentLoss(nn.Module):
    def __init__(self, content_feature):
        super(ContentLoss, self).__init__()
        self.content_feature = content_feature.detach()

    def forward(self, input):
        self.loss = torch.nn.functional.mse_loss(input, self.content_feature)
        return input

# Style Loss Class
class StyleLoss(nn.Module):
    def __init__(self, style_feature):
        super(StyleLoss, self).__init__()
        self.style_feature = StyleLoss.gram_matrix(style_feature).detach()

    @staticmethod
    def gram_matrix(tensor):
        _, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = torch.nn.functional.mse_loss(G, self.style_feature)
        return input


def setup_model(model, content_img, style_img, content_layers, style_layers):
    model = model.features.eval() # Extract features layers # Set to evaluation mode
    for param in model.parameters():
        param.requires_grad = False

    new_model = nn.Sequential()
    new_model.content_losses = []
    new_model.style_losses = []
    i = 0
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
            layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)  # Replace MaxPool2d with AvgPool2d
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        new_model.add_module(name, layer)

        if name in content_layers:
            content_feature = new_model(content_img).detach()
            content_loss = ContentLoss(content_feature)
            new_model.add_module(f'content_loss_{i}', content_loss)
            new_model.content_losses.append(content_loss)

        if name in style_layers:
            style_feature = new_model(style_img).detach()
            style_loss = StyleLoss(style_feature)
            new_model.add_module(f'style_loss_{i}', style_loss)
            new_model.style_losses.append(style_loss)

    return new_model

# Image optimization function
def train_image(input_img, model, num_steps, style_weight, content_weight, save_images):
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    loss_history = {
        "total_loss": [],
        "content_loss": [],
        "style_loss": [],
        "steps": []
    }
    def closure():
        optimizer.zero_grad()
        model(input_img)

        content_score = sum(cl.loss for cl in model.content_losses)
        style_score = sum(sl.loss for sl in model.style_losses)

        loss = content_weight * content_score + style_weight * style_score
        loss.backward()

        if run[0] % 4 == 0:
            loss_history["total_loss"].append(loss.item())
            loss_history["content_loss"].append(content_score.item())
            loss_history["style_loss"].append(style_score.item())
            loss_history["steps"].append(run[0])

            if save_images:
                plt.imsave(f"output_images/output_{run[0]}.png", deprocess_image(input_img))

        run[0] += 1
        return loss

    for _ in tqdm(range(num_steps),desc="Optimizing Image"):
        optimizer.step(closure)
        # Clamp the image pixel values to 0-1
        with torch.no_grad():
            input_img.clamp_(0, 1)

    return input_img, loss_history

def transfer_style(content_img, style_img, input_img=None, model=vgg19(weights=VGG19_Weights.DEFAULT), num_steps=20, 
                    style_weight=1000, content_weight=1,
                    content_layers=['conv_10'], 
                    style_layers=['conv_2', 'conv_3', 'conv_5', 'conv_9', 'conv_13'],
                    save_images=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images (with normalization)
    content_img = load_image(content_img).to(device)
    style_img = load_image(style_img, shape=content_img.shape[-2:]).to(device)
    
    # Initialize input image
    if input_img is None:
        input_img = torch.rand_like(content_img.data, requires_grad=True)
    else:
        input_img = load_image(input_img, shape=content_img.shape[-2:]).to(device)
        input_img = input_img.requires_grad_(True)
    
    # Create directory to save images
    os.makedirs("output_images", exist_ok=True)

    # Setup model
    model = setup_model(model.to(device), content_img, style_img, 
                        content_layers, style_layers)
    
    # Run style transfer
    output_img, loss_history = train_image(input_img, model, num_steps, style_weight, content_weight, save_images)
    output_img = deprocess_image(output_img)

    return output_img, loss_history