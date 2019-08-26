#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable

IMAGE_SIZE = 228


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

    return (e.data.numpy().squeeze().tolist() for e in topk)


'''
Input arguments
  num_classes: number of classes in the dataset.
               This is equal to the number of output neurons.
'''


def initialize_alexnet(num_classes):
    # load the pre-trained Alexnet
    alexnet = torchvision.models.alexnet(pretrained=True)

    # get the number of neurons in the penultimate layer
    in_features = alexnet.classifier[6].in_features

    # re-initalize the output layer
    alexnet.classifier[6] = torch.nn.Linear(in_features=in_features,
                                            out_features=num_classes)

    return alexnet


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

    return cumulative_loss/samples, cumulative_accuracy/samples*100


def train(net,data_loader, optimizer, cost_function, device='cuda:0'):
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
        loss = cost_function(outputs,targets)

        # Reset the optimizer

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        optimizer.zero_grad()

        # Better print something, no?
        samples+=inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

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
         plot_name='alexnet_sgd',
         img_root=None):
    train_loader, test_loader = get_data(batch_size=batch_size,
                                         img_root=img_root)
    net = initialize_alexnet(num_classes=num_classes).to(device)

    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)

    cost_function = get_cost_function()

    train_loss_curve = []
    train_accuracy_curve = []
    test_loss_curve = []
    test_accuracy_curve = []

    print('Before training:')
    train_loss, train_accuracy = test(net, train_loader, cost_function)
    test_loss, test_accuracy = test(net, test_loader, cost_function)

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    for e in range(epochs):
        train_loss, train_accuracy = train(net, train_loader, optimizer, cost_function)
        test_loss, test_accuracy = test(net, test_loader, cost_function)
        print('Epoch: {:d}'.format(e+1))
        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
        print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
        print('-----------------------------------------------------')

        train_loss_curve.append(train_loss)
        train_accuracy_curve.append(train_accuracy)
        test_loss_curve.append(test_loss)
        test_accuracy_curve.append(test_accuracy)

        # Add values to plots
        plt.clf()
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title('Loss curves')
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('loss')
        axs[0].plot(train_loss_curve, label='Train')
        axs[0].plot(test_loss_curve, label='Test')

        axs[1].set_title('Accuracy curves')
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('accuracy')
        axs[1].plot(train_accuracy_curve, label='Train')
        axs[1].plot(test_accuracy_curve, label='Test')
        plt.legend()
        plt.savefig("curve_" + plot_name + ".pdf")

    print('After training:')
    train_loss, train_accuracy = test(net, train_loader, cost_function)
    test_loss, test_accuracy = test(net, test_loader, cost_function)

    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    probs, classes = predict('data/InputRec/image.png', topk=2, net=net)
    print(probs)
    print(classes)


# num_classes = num_faces_recognited + 1 (no rec)
main(plot_name='alexnet',
     img_root='data/ProfilePhotos', num_classes=4, epochs=3)

