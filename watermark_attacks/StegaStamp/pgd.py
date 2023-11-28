# uskarim

import os
import random
import glob
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.utils
from torchvision import transforms
import clip
import numpy as np
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# from utils import CustomImageFolder
import torchvision.models as models
from tqdm import tqdm
import torch.cuda.amp as amp


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None, data_cnt=-1, y=0):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*/*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*/*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*/*.JPEG")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*/*.jpg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.png")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.JPEG")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        random.seed(17)
        random.shuffle(self.filenames)
        if data_cnt != -1:
            self.filenames = self.filenames[:data_cnt]
        if data_dir[-1] != '/':
            data_dir += '/'
        self.img_ids = [x.replace(data_dir, '').replace('/', '_') for x in self.filenames]
        self.transform = transform
        self.y = y

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y, filename

    def __len__(self):
        return len(self.filenames)



def pgd_attack(model, criterion, images, labels, eps=8/255, alpha=None, iters=10, norm=None, noise=None):
    # def pgd_attack(model, criterion, images, labels, eps=0.007, alpha=0.01, iters=10, norm=None, noise=None):
    # def pgd_attack(model, criterion, images, labels, eps=0.0039, alpha=0.01, iters=10, norm=None, noise=None):
    if alpha == None:
        alpha = eps * 0.05
    original_image = images.clone().detach()
    if noise is None:
        noise = torch.rand_like(images) * 2 * eps - eps
    m_images = images + noise
    images = torch.clamp(m_images, 0, 1).detach().requires_grad_(True)
    original_noise = noise
    for k in range(iters):
        # if k == 0:
        #     while True:
        #         if noise is None:
        #             noise = torch.rand_like(images) * 2 * eps - eps
        #         images = images.clone().detach().requires_grad_(True)
        #         # print(images.shape, noise.shape)
        #         m_images = images + noise
        #         outputs = model(norm(m_images))[:, 0]
        #         loss = criterion(outputs, labels)
        #         # if loss > 0.0001:
        #         if loss > 0.:
        #             images = m_images.clone().detach().requires_grad_(True)
        #             break
        # else:
        #   images = images.clone().detach().requires_grad_(True)

        outputs = model(norm(images))[:, 0]
        loss = criterion(outputs, labels) # - 0.1 * (images - original_image).pow(2).sum().sqrt()
        loss.backward()

        
        # if k % 20 == 0:
        #     print(loss.item())

        # PGD step to perturb the images
        grad = images.grad.data
        images = images + alpha * grad.sign()

        # Project the perturbation back to the epsilon ball
        diff = images - original_image
        diff = torch.clamp(diff, -eps, eps)
        images = torch.clamp(original_image + diff, 0, 1).detach().requires_grad_(True)
    # import pdb ; pdb.set_trace()

    images = images.detach()
    ret_noise = images - original_image
    # if original_noise is not None:
    #     ret_noise = torch.cat((ret_noise, noise[images.shape[0]:batch_size]))
    #     print(ret_noise.shape)
    # return images, images - original_image
    return images, ret_noise


def random_attack(model, criterion, images, labels, eps=0.03, alpha=0.01, iters=10, norm=None, noise=None):
    # def pgd_attack(model, criterion, images, labels, eps=0.007, alpha=0.01, iters=10, norm=None, noise=None):
    # def pgd_attack(model, criterion, images, labels, eps=0.0039, alpha=0.01, iters=10, norm=None, noise=None):
    noise = torch.rand_like(images) * 2 * eps - eps
    images = images + noise
    images = torch.clamp(images, 0, 1).detach()
    return images, noise


def load_dataset(transform, path, y=1):
    dataset = CustomImageFolder(path, transform=transform, y=y, data_cnt=64)
    N = len(dataset)
    print(f'number of watermarked images --> {N}')

    return dataset


batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define a binary classification model using ResNet18
class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))


# Function to train the model
def eval_model(model, val_dataloader, norm, num_epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.BCELoss()
    # Evaluation on the validation set
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, file_names in tqdm(val_dataloader):
            labels = labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_normalized = norm(inputs)
            outputs = model(inputs_normalized)[:, 0]
            val_loss += criterion(outputs, labels).item()
            predicted = torch.round(outputs)
            file_names = [filename.split('/')[-1] for filename in file_names]
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


def eval_model2(model, inputs, labels, norm, num_epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.BCELoss()
    # Evaluation on the validation set
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        labels = labels.type(torch.FloatTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_normalized = norm(inputs)
        outputs = model(inputs_normalized)[:, 0]
        val_loss += criterion(outputs, labels).item()
        predicted = torch.round(outputs)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


# Main function to run the binary classification task
def main(method, mode='wm', eps=8):
    # Create the dataset and split it into train and validation sets
    _, preprocess = clip.load('ViT-L/14', device)
    pre_norm = torchvision.transforms.Compose(preprocess.transforms[:-1])
    # norm = preprocess.transforms[-1]
    if method == 'stegaStamp':
        norm = preprocess.transforms[-1]
    else:
        norm = (lambda x: x)
    print('loading dataset ...')
    print(f"method: {method}, mode: {mode}, eps: {eps}")
    if mode == 'wm':
        if method == 'stegaStamp':
            test_set = load_dataset(pre_norm, "./out", y=1)
        elif method == 'treeRing':
            test_set = load_dataset(pre_norm, '../images/test/treeRing', y=1)
    else:
        if method == 'stegaStamp':
            test_set = load_dataset(pre_norm, "path_to_image_dataset", y=0)
        elif method == 'treeRing':
            test_set = load_dataset(pre_norm, '../images/imagenet/treeRing_no_wm', y=0)
            
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)  # check

    # Create the model
    model = ResNetBinaryClassifier()
    cp = torch.load(f'./models/classifier_{method}.pt')
    load_pretrained = True
    if load_pretrained:
        model.load_state_dict(cp['model'])
    model.eval()
    eval_model(model, val_loader, norm)
    model.requires_grad_(False)
    model = model.to(device)
    criterion = nn.BCELoss()
    images_all = []
    labels_all = []
    diffs_all = []
    file_names_all = []
    noise = None
    j_num = 2
    warm_up = 10
    for j in range(j_num):
        for i, (image, target, filenames) in tqdm(enumerate(val_loader)):
            target = target.type(torch.FloatTensor)
            image, target = image.cuda(), target.cuda()
            # image = norm(image)
            filenames = [filename.split('/')[-1] for filename in filenames]
            images, noise = pgd_attack(model, criterion, image, target, iters=300, norm=norm, noise=noise, eps=eps/255)
            # images, noise = pgd_attack_l2(model, criterion, image, target, iters=500, norm=norm, noise=noise)
            # print(filenames)
            if j < j_num - 1 and i > warm_up:
                break
            if j == j_num - 1:
                images_all.append(images)
                labels_all.append(target)
                diffs_all.append(images - image)
                file_names_all.extend(filenames)
    images_all = torch.cat(images_all, dim=0)
    diffs_all = torch.cat(diffs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    if mode == 'wm':
        images_dir = f'./test_images/adv_images_{method}_{eps}/'
        diffs_dir = f'./test_images/diff_images_{method}_{eps}/'
    else:
        images_dir = f'./test_images/adv_org_images_{method}_{eps}/'
        diffs_dir = f'./test_images/diff_org_images_{method}_{eps}/'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(diffs_dir, exist_ok=True)
    for i, (image, diff, file_name) in enumerate(zip(images_all, diffs_all, file_names_all)):
        torchvision.utils.save_image(image, os.path.join(images_dir, file_name))
        torchvision.utils.save_image(diff, os.path.join(diffs_dir, file_name), normalize=True, scale_each=True)
        # torchvision.utils.save_image(diff, os.path.join(diffs_dir, file_name))

    # adv_test_set = load_dataset(preprocess, images_dir)
    # todo: why this is not working?
    # adv_val_loader = torch.utils.data.DataLoader(adv_test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    eval_model2(model, images_all, labels_all, norm)


# Run the main function if this script is executed
if __name__ == "__main__":
    # main(mode='wm', eps=14)
    # main(mode='org', eps=14)
    # main(method='treeRing', mode='wm', eps=8)
    # main(method='treeRing', mode='org', eps=8)
    # main(method='stegaStamp', mode='wm', eps=12)
    # main(method='stegaStamp', mode='org', eps=12)
    # exit()
    # main(method='treeRing', mode='wm', eps=4)
    # main(method='treeRing', mode='org', eps=4)
    # main(method='treeRing', mode='wm', eps=2)
    # main(method='treeRing', mode='org', eps=2)
    # main(method='treeRing', mode='wm', eps=10)
    # main(method='treeRing', mode='org', eps=10)
    # main(method='treeRing', mode='wm', eps=12)
    # main(method='treeRing', mode='org', eps=12)
    # main(method='treeRing', mode='wm', eps=6)
    # main(method='treeRing', mode='org', eps=6)
    # main(method='treeRing', mode='wm', eps=14)
    # main(method='treeRing', mode='org', eps=14)
    main(method='treeRing', mode='wm', eps=4)
    # main(method='stegaStamp', mode='org', eps=14)
    # main(mode='wm', eps=12)
    # main(mode='org', eps=12)
    # main(mode='wm', eps=13)
    # main(mode='org', eps=13)
    # main(mode='wm', eps=10)
    # main(mode='org', eps=10)
