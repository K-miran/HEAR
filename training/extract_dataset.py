import argparse
from pathlib import Path

import numpy as np
import torch
import pickle
from HEnet import HENet
from HEnet2D import HENet2D
from sampler import Sampler
from torch import nn, optim
from torch.autograd import Variable
from sampler import Sampler
from dataset import FallData, Flip, Scale, Translate
from dataset import Normalize, OutsideIamge, RemoveJoints
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.transforms import transforms
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='python extract_dataset.py')
    parser.add_argument('model', type=str, help='Path to the saved model')
    return parser.parse_args()


def main(use2d=False, model_name="1d_model", filters=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('TrainAndTestData/inputs_test.pkl', 'rb') as source:
        test_inputs = pickle.load(source)
    with open('TrainAndTestData/label_test.pkl', 'rb') as source:
        test_labels = pickle.load(source)
        
    test_inputs = np.reshape(test_inputs, (test_inputs.shape[0], 32, 15, 3))
    test_inputs = np.transpose(test_inputs, (0, 3, 1, 2))
    test_inputs[:, 2, :, :] = 0
    test_labels = np.asarray(test_labels, dtype=np.int32)
    test_dataset = FallData(test_inputs, test_labels,
                            transform=transforms.Compose([
                                RemoveJoints(),
                                Normalize()
                            ]))
    torch.manual_seed(0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    Encrypted_model = HENet2D(10, filters) if (use2d) else HENet(10, filters)
    Encrypted_model.load_state_dict(torch.load(f"{model_name}"))
    Encrypted_model.to(device)
    Encrypted_model.eval()
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    acc = 0
    test_input = pd.DataFrame()
    test_labels = pd.DataFrame()
    test_prob = pd.DataFrame()
    test_performance = pd.DataFrame()
    
    for i, data in enumerate(test_loader):
        test_input = test_input.append(pd.DataFrame(data["x"].numpy().flatten()).T)
        test_labels = test_labels.append(pd.DataFrame(data["y"].numpy()))
        test_data = Variable(data["x"]).to(device)
        test_label = data["y"].numpy()[0]
        outputs = Encrypted_model(test_data).data.cpu().numpy()[0]
        test_prob = test_prob.append(pd.DataFrame(outputs).T)
        predicts = np.argmax(outputs)
        acc += (predicts == test_label).sum()
        if test_label == 9:
            if predicts == 9:
                TP += 1
            else:
                FN += 1
        elif test_label != 9:
            if predicts == 9:
                FP += 1
            else:
                TN += 1

    accuracy = acc / len(test_loader)

    print(test_input.shape, test_labels.shape, test_prob.shape)

    Fall_sensitivity = TP / (TP + FN)
    Fall_specificity = TN / (TN + FP)
    Fall_precision = TP / (TP + FP)
    Fall_F1 = 2 * Fall_precision * Fall_sensitivity / (Fall_precision + Fall_sensitivity)
    test_performance = test_performance.append(
        pd.DataFrame(np.array([Fall_sensitivity, Fall_specificity, Fall_precision, accuracy, Fall_F1])).T)
    test_input.to_csv("../dataset/test_input.csv", index=False, header=False)
    test_labels.to_csv("../dataset/test_label.csv", index=False, header=False)
    test_prob.to_csv(f"../dataset/test_prob_conv{'2d' if use2d else '1d'}_{filters}.csv", index=False, header=False)
    test_performance.to_csv(f"../dataset/test_performance_conv{'2d' if use2d else '1d'}_{filters}.csv", index=False,
                            header=False)

    print('\33[34m' + "Test set accuracy: %.2f, Fall sensitivity: %.2f" % (accuracy, Fall_sensitivity
                                                                           ) + '\033[0m')
    return accuracy, Fall_sensitivity


if __name__ == '__main__':
    args = parse_args()
    filename = Path(args.model).name.title()
    main(use2d=True if "2d" in args.model else False, model_name=args.model, filters=int(filename.split("_")[2]))
