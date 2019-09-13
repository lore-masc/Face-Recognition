# coding: utf-8
import argparse
import operator
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.autograd import Variable

IMAGE_SIZE = 228


def get_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    return classes


'''
Input arguments:
  batch_size: mini batch size used during training
  img_root: path to the dataset parent folder. 
            The folder just above the sub-folders or class folders
'''


def get_data(batch_size, img_root):
    # Prepare data transformations for the train loader
    transform = list()
    transform.append(T.Resize((IMAGE_SIZE, IMAGE_SIZE)))              # Resize each PIL image
    transform.append(T.ToTensor())                                    # converts Numpy to Pytorch Tensor
    transform.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))          # Normalize with ImageNet mean
    transform = T.Compose(transform)                                  # Composes the above transformations into one.

    # Load data
    dataset = torchvision.datasets.ImageFolder(root=img_root, transform=transform)

    # Create train and test splits
    # We will create a 80:20 % train:test split
    num_samples = len(dataset)
    training_samples = int(num_samples * 0.8 + 1)
    test_samples = num_samples - training_samples

    training_data, test_data = torch.utils.data.random_split(dataset,
                                                             [training_samples, test_samples])

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader


'''
Input arguments
  image_path: path of the image to predict
  net: specify neural network
  device: gpu device name in order to compute net
'''


def predict(image_path, net, topk=2, device='cuda:0'):
    preprocess = T.transforms.Compose([
        T.transforms.Resize(IMAGE_SIZE, IMAGE_SIZE),
        T.transforms.ToTensor(),
        T.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img = preprocess(img)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    inputs = Variable(img).to(device)
    logits = net.forward(inputs)
    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)
    torch.cuda.empty_cache()

    return (e.data.numpy().squeeze().tolist() for e in topk)


'''
Input arguments
  num_classes: number of classes in the dataset.
               This is equal to the number of output neurons.
'''


def initialize_net(num_classes):
    if Path(MODEL_PATH).exists():
        net = torch.load(MODEL_PATH)
    else:
        # load the pre-trained selected net
        net = torchvision.models.googlenet(pretrained=True)
        # get the number of neurons in the penultimate layer
        in_features = net.fc.in_features
        # re-initalize the output layer
        net.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return net


def get_cost_function():
    cost_function = torch.nn.CrossEntropyLoss()
    return cost_function


def get_optimizer(model, lr, wd, momentum):
    # we will create two groups of weights, one for the newly initialized layer
    # and the other for rest of the layers of the network

    final_layer_weights = []
    rest_of_the_net_weights = []

    # we will iterate through the layers of the network
    for name, param in model.named_parameters():
        if name.startswith('classifier.6'):
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    # so now we have divided the network weights into two groups.
    # We will train the final_layer_weights with learning_rate = lr
    # and rest_of_the_net_weights with learning_rate = lr / 10

    optimizer = torch.optim.SGD([
        {'params': rest_of_the_net_weights},
        {'params': final_layer_weights, 'lr': lr}
    ], lr=lr / 10, weight_decay=wd, momentum=momentum)

    return optimizer


def test(net, data_loader, cost_function, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.eval()  # Strictly needed if network contains layers which has different behaviours between train and test
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)

            # Apply the loss
            loss = cost_function(outputs, targets)

            # Better print something
            samples += inputs.shape[0]
            cumulative_loss += loss.item()  # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

            # Free cuda memory
            torch.cuda.empty_cache()

    return cumulative_loss/samples, cumulative_accuracy/samples*100


def train(net, data_loader, optimizer, cost_function, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.

    net.train()  # Strictly needed if network contains layers which has different behaviours between train and test
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)

        # Apply the loss
        loss = cost_function(outputs, targets)

        # Reset the optimizer

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        optimizer.zero_grad()

        # Better print something, no?
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

        # Free cuda memory
        torch.cuda.empty_cache()

    return cumulative_loss/samples, cumulative_accuracy/samples*100


'''
Input arguments
  batch_size: Size of a mini-batch
  device: GPU where you want to train your network
  weight_decay: Weight decay co-efficient for regularization of weights
  momentum: Momentum for SGD optimizer
  epochs: Number of epochs for training the network
  num_classes: Number of classes in your dataset
  visualization_name: Name of the visualization folder
  img_root: The root folder of images
'''


def main(batch_size=128, 
         device='cuda:0', 
         learning_rate=0.001, 
         weight_decay=0.000001, 
         momentum=0.9, 
         epochs=50, 
         num_classes=10,
         img_root=None,
         save=True,
         perform_training=True):

    train_loader, test_loader = get_data(batch_size=batch_size,
                                         img_root=img_root)

    net = initialize_net(num_classes=num_classes).to(device)

    print("{")

    if perform_training:
        optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

        cost_function = get_cost_function()

        train_loss_curve = []
        train_accuracy_curve = []
        test_loss_curve = []
        test_accuracy_curve = []

        print('\t"training": [')
        train_loss, train_accuracy = test(net, train_loader, cost_function, device)
        test_loss, test_accuracy = test(net, test_loader, cost_function, device)

        print('\t\t{"Training loss": %.5f, "Training accuracy": %.2f},' % (train_loss, train_accuracy))
        print('\t\t{"Test loss": %.5f, "Test accuracy": %.2f},' % (test_loss, test_accuracy))
        # print('-----------------------------------------------------')

        for e in range(epochs):
            train_loss, train_accuracy = train(net, train_loader, optimizer, cost_function, device)
            test_loss, test_accuracy = test(net, test_loader, cost_function, device)
            # print('Epoch: {:d}'.format(e+1))
            print('\t\t{"Training loss": %.5f, "Training accuracy": %.2f},' % (train_loss, train_accuracy))
            print('\t\t{"Test loss": %.5f, "Test accuracy": %.2f},' % (test_loss, test_accuracy))
            # print('-----------------------------------------------------')

            train_loss_curve.append(train_loss)
            train_accuracy_curve.append(train_accuracy)
            test_loss_curve.append(test_loss)
            test_accuracy_curve.append(test_accuracy)

        train_loss, train_accuracy = test(net, train_loader, cost_function, device)
        test_loss, test_accuracy = test(net, test_loader, cost_function, device)

        print('\t\t{"Training loss": %.5f, "Training accuracy": %.2f},' % (train_loss, train_accuracy))
        print('\t\t{"Test loss": %.5f, "Test accuracy": %.2f}' % (test_loss, test_accuracy))
        # print('-----------------------------------------------------')

        print('\t],')
        # saving model
        if save:
            # print("Saving weights")
            torch.save(net, MODEL_PATH)

    # multi prediction
    dict = {}
    for k in get_classes(img_root):
        dict.update({k: 0})

    print('\t"prediction": {')
    print('\t\t"files": [')
    if len(os.listdir(INPUTS_PATH)) > 0:
        for filename in os.listdir(INPUTS_PATH):
            if filename != os.listdir(INPUTS_PATH)[0]:
                print(',')
            print('\t\t\t{')
            print('\t\t\t\t"name": "' + filename + '",')
            probs, classes = predict(INPUTS_PATH + filename, topk=num_classes, net=net, device=device)
            matrix = np.c_[classes, probs]
            matrix = matrix[np.argsort(matrix[:, 0])]
            for c in matrix[:, 0]:
                dict[get_classes(img_root)[int(c)]] += matrix[int(c), 1]
            print('\t\t\t\t"probs": ' + str(["{0:0.3f}".format(i) for i in probs]) + ',')
            print('\t\t\t\t"classes": ' + str([get_classes(img_root)[int(c)] for c in classes]))
            print('\t\t\t}', end='')
        print()
        print('\t\t],')

        new_dict = {k: v / len(os.listdir(INPUTS_PATH)) for k, v in dict.items()}
        print('\t\t"average": ' + str(new_dict) + ',')
        print('\t\t"result": ', end='')
        if gini(new_dict.values()) < 0.5:
            print('other')
        else:
            most_prob = max(new_dict.items(), key=operator.itemgetter(1))
            print('"' + str(most_prob[0]) + '"')
    print('\t}')
    print('}')


def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


parser = argparse.ArgumentParser(description='Process face recognition using GoogLeNet.')
parser.add_argument('-d', '--dataset', action='store', dest='dataset', help='write the dataset relative path', required=True)
parser.add_argument('-i', '--inputs', action='store', dest='inputs', help='write the inputs relative path', required=True)
parser.add_argument('-w', '--weights', action='store', dest='weights', help='write the model relative path', required=True)
parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, help='write the number of epochs to run during the train', default=3)
parser.add_argument('-b', '--batch', action='store', dest='batch_size', type=int, help='write the batch dimension', default=16)
parser.add_argument('-t', '--training', action='store_true', dest='training', help='set option in order to perform training before predictions')
parser.add_argument('-s', '--save', action='store_true', dest='save', help='set option in order to save new weights')
parser.add_argument('-g', '--device', action='store', dest='device', help='set device processor name (cpu or cuda:0)', default='cpu')

args = parser.parse_args()

MODEL_PATH = args.weights     # "model/weights"
DATASET_PATH = args.dataset   # "data/ProfilePhotos/"
INPUTS_PATH = args.inputs     # "data/InputRec/"

main(img_root=DATASET_PATH,
     num_classes=len(os.listdir(DATASET_PATH)), epochs=args.epochs, batch_size=args.batch_size,
     perform_training=args.training, save=args.save, device=args.device)
