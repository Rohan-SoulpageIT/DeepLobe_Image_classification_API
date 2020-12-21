# Make prediction
# transformations on the data.
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def pre_processing(img_bytes):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = preprocess(Image.open(img_bytes).convert("RGB"))
    return torch.unsqueeze(img_t, 0)


def classifiy_image(img_data):
    # loading the resnet50 model
    resnet = models.resnet18(pretrained=True)
    # getting model for prediction.
    resnet.eval()
    # inference we are getting in form of the tensors
    out = resnet(img_data)
    # getting labels from txt files.
    with open("image_net_classes.txt", "r") as fp:
        labels = [line.strip() for line in fp.readlines()]
    _, index = torch.max(out, 1)

    # sorting the results.
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    class_list = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

    # returning the class
    return str(class_list[0][0])