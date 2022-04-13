import argparse

import numpy as np
import torch
import pickle
from HEnet import HENet
from HEnet2D import HENet2D
from sampler import Sampler
from torch import nn, optim
from torch.autograd import Variable
from sampler import Sampler
from pathlib import Path
from dataset import FallData, Flip, Scale, Translate
from dataset import Normalize, OutsideIamge, RemoveJoints
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import time
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="python train.py")
    parser.add_argument('--filters', metavar='NUM', type=int,
                        default=128,
                        help='number of filters for the first convolutional layer \
                         (default: %(default)s)')
    parser.add_argument('--type', metavar='STR', type=str,
                        default='2d',
                        help='CNN network type \
                             (default: %(default)s)')
    return parser.parse_args()


def main(use2d=False, name="1d_model", filters=128):
    Path("saved_model").mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 100
    test_step = 50
    with open('TrainAndTestData/inputs.pkl', 'rb') as source:
        inputs = pickle.load(source)
    with open('TrainAndTestData/label.pkl', 'rb') as source:
        labels = pickle.load(source)
    with open('TrainAndTestData/inputs_test.pkl', 'rb') as source:
        test_inputs = pickle.load(source)
    with open('TrainAndTestData/label_test.pkl', 'rb') as source:
        test_labels = pickle.load(source)
        
    inputs = np.reshape(inputs, (inputs.shape[0], 32, 15, 3))
    inputs = np.transpose(inputs, (0, 3, 1, 2))
    inputs[:, 2, :, :] = 0
    labels = np.asarray(labels, dtype=np.int32)
    test_inputs = np.reshape(test_inputs, (test_inputs.shape[0], 32, 15, 3))
    test_inputs = np.transpose(test_inputs, (0, 3, 1, 2))
    test_inputs[:, 2, :, :] = 0
    test_labels = np.asarray(test_labels, dtype=np.int32)
    train_dataset = FallData(inputs, labels,
                             transform=transforms.Compose([
                                 OutsideIamge(),
                                 RemoveJoints(),
                                 Scale(),
                                 Translate(),
                                 Flip(),
                                 Normalize()
                             ]))
    test_dataset = FallData(test_inputs, test_labels,
                            transform=transforms.Compose([
                                RemoveJoints(),
                                Normalize()
                            ]))

    datasampler = Sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=datasampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    Encrypted_model = HENet2D(10, filters) if use2d else HENet(10, filters)
    Encrypted_model.to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Encrypted_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_cost = 100
    
    for epoch in range(num_epochs):
        costs = 0
        Encrypted_model.train()
        scheduler.step()
        for i, data in enumerate(train_loader):
            train_data = Variable(data["x"]).to(device)
            train_label = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
            optimizer.zero_grad()
            outputs = Encrypted_model(train_data)
            cost = CrossEntropyLoss(outputs, train_label)
            cost.backward()
            costs += cost.item()
            optimizer.step()
        total_cost = costs / len(train_loader)
        if total_cost < best_cost:
            print('\033[91m' + "Saving the model" + '\033[0m')
            torch.save(Encrypted_model.state_dict(), f"saved_model/{name}.torch")
            best_cost = total_cost
        print("Epoch: %s, Loss: %s" % (epoch, total_cost))
        acc = 0
        if ((epoch + 1) % test_step == 0):
            Encrypted_model.eval()
            for i, data in enumerate(test_loader):
                test_data = Variable(data["x"]).to(device)
                test_label = data["y"].numpy()[0]
                outputs = Encrypted_model(test_data).data.cpu().numpy()[0]
                predicts = np.argmax(outputs)
                acc += (predicts == test_label).sum()
            accuracy = acc / len(test_loader)
            print('\33[34m' + "Test set accuracy: %.2f" % (accuracy) + '\033[0m')
    print('\033[91m' + "Saving the final model to saved_model/" + name + "_final.torch" + '\033[0m')
    torch.save(Encrypted_model.state_dict(), f"saved_model/{name}_final.torch")


if __name__ == '__main__':
    args = parse_args()
    main(filters=args.filters, use2d=True if args.type == '2d' else False, name=f"{args.type}_model_{args.filters}")
