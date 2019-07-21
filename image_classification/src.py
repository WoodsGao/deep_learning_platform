import cv2
import os
import shutil
import time
import torch
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from PIL import Image
import numpy as np
from threading import Thread
from .models import Dataset, Classification, ImageData

size = (224, 224)
data_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:%s' % device)
if not os.path.exists('./cnn_checkpoints'):
    os.mkdir('./cnn_checkpoints')
if not os.path.exists('./cnn_tmp'):
    os.mkdir('./cnn_tmp')


def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    cv2.imwrite(image_path, img)


def create_cnn_model(dataset_id):
    if dataset_id is None:
        return -1
    dataset = Dataset.objects.get(id=dataset_id)
    classifications = Classification.objects.filter(dataset_id=dataset_id)
    num_classes = classifications.count()
    net = models.densenet201(pretrained=True)
    channel_in = net.classifier.in_features
    net.classifier = nn.Sequential(
        nn.BatchNorm1d(channel_in),
        nn.Dropout(0.5),
        nn.Linear(channel_in, num_classes)
    )
    net = net.to(device)
    state = {
        'net': net.state_dict(),
        'acc': 0,
        'loss': 1000,
        'epoch': 0
    }
    torch.save(state, './cnn_checkpoints/%s.t7' % dataset_id)
    print('create new')
    dataset.loss = 1000.0
    dataset.acc = 0.0
    dataset.now_epoches = 0
    dataset.target_epoches = - dataset.target_epoches
    dataset.save()
    print('saved')


def validate_image(img_data):
    dataset = img_data.dataset
    classifications = Classification.objects.filter(dataset_id=dataset.id)
    img = cv2.imread(img_data.img.path)
    num_classes = classifications.count()
    net = models.densenet201(pretrained=True)
    channel_in = net.classifier.in_features
    net.classifier = nn.Sequential(
        nn.BatchNorm1d(channel_in),
        nn.Dropout(0.5),
        nn.Linear(channel_in, num_classes)
    )
    checkpoint = torch.load('./cnn_checkpoints/%s.t7' %
                            dataset.id, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    net.eval()
    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = Image.fromarray(input_data)
    input_data = data_transforms(input_data)
    input_data = input_data.to(device)
    input_data = torch.unsqueeze(input_data, 0)
    output_data = net(input_data)
    output_data = np.array(output_data.cpu().data)[0]
    classification = classifications[int(np.argmax(output_data))]
    if img_data.save_to_db:
        image = ImageData()
        image.img = img_data.img
        image.classification = classification
        image.save()
    return {'classification_id': classification.id, 'classification_name': classification.name,
            'dataset_id': classification.dataset.id, 'dataset_name': classification.dataset.name}


def train_model_thread():
    while True:
        datasets = Dataset.objects.all()
        for dataset in datasets:
            try:
                if dataset.target_epoches == 0:
                    continue
                if dataset.target_epoches < 0:
                    create_cnn_model(dataset.id)
                    continue
                if dataset.target_epoches <= dataset.now_epoches:
                    dataset.target_epoches = 0
                    dataset.now_epoches = 0
                    dataset.save()
                    continue
                dataset.now_epoches += 1
                dataset.save()
                # load model
                classifications = Classification.objects.filter(
                    dataset=dataset)
                num_classes = classifications.count()
                net = models.densenet201(pretrained=True)
                channel_in = net.classifier.in_features
                net.classifier = nn.Sequential(
                    nn.BatchNorm1d(channel_in),
                    nn.Dropout(0.5),
                    nn.Linear(channel_in, num_classes)
                )
                checkpoint = torch.load(
                    './cnn_checkpoints/%s.t7' % dataset.id, map_location=device)
                net.load_state_dict(checkpoint['net'])
                net = net.to(device)
                best_acc = checkpoint['acc']
                best_loss = checkpoint['loss']
                epoch = checkpoint['epoch']
                criterion = nn.CrossEntropyLoss()
                ignored_params = list(map(id, net.classifier.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params,
                                     net.parameters())
                optimizer = optim.SGD([
                    {'params': base_params},
                    {'params': net.classifier.parameters(), 'lr': 1e-2}
                ], lr=dataset.lr)
                # create dataset
                train_dir = os.path.join('./cnn_tmp', 'train')
                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir)
                os.mkdir(train_dir)
                val_dir = os.path.join('./cnn_tmp', 'val')
                if os.path.exists(val_dir):
                    shutil.rmtree(val_dir)
                os.mkdir(val_dir)
                for c in classifications.all():
                    os.mkdir(os.path.join(train_dir, str(c.id)))
                    os.mkdir(os.path.join(val_dir, str(c.id)))
                    images = ImageData.objects.filter(classification=c).all()
                    for i, image in enumerate(images):
                        if i % 3 != 0:
                            shutil.copy(image.img.path, os.path.join(
                                train_dir, str(c.id)))
                        else:
                            shutil.copy(images[i].img.path,
                                        os.path.join(val_dir, str(c.id)))
                train_image = ImageFolder(train_dir, data_transforms)
                train_loader = torch.utils.data.DataLoader(
                    train_image, batch_size=32, shuffle=True, num_workers=8)
                val_image = ImageFolder(val_dir, data_transforms)
                val_loader = torch.utils.data.DataLoader(
                    val_image, batch_size=32, shuffle=True, num_workers=8)
                # train
                net.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                # validate
                net.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        predicted = outputs.max(1)[1]
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                # Save checkpoint.
                acc = 100. * correct / total
                loss = val_loss / (batch_idx + 1)
                epoch += 1
                if acc > best_acc:
                    print('\nSaving..')
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'loss': loss,
                        'epoch': epoch
                    }
                    torch.save(state, './cnn_checkpoints/%s.t7' % dataset.id)
                    best_loss = loss
                    best_acc = acc
                    dataset = Dataset.objects.get(id=dataset.id)
                    if dataset.target_epoches >= 0:
                        dataset.loss = loss
                        dataset.acc = acc
                        dataset.save()
            except Exception as e:
                print(e)
        time.sleep(1)


def train_model():
    t = Thread(target=train_model_thread)
    t.setDaemon(True)
    t.start()


train_model()
