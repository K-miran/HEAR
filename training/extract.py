import argparse
import pandas as pd
from HEnet import HENet
from HEnet2D import HENet2D
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='python extract.py')
    parser.add_argument('model', type=str, help='Path to the saved model')
    return parser.parse_args()


def main(model_name='1d_model', use2d=False, filters=128):
    folder = f"../dataset/parameters_conv{'2d' if use2d else '1d'}_{filters}"
    Encrypted_model = HENet2D(10, filters) if use2d else HENet(10, filters)
    Encrypted_model.load_state_dict(torch.load(model_name))
    Encrypted_model.to(device)
    Path(folder).mkdir(exist_ok=True, parents=True)
    print(f"Extracting to {folder}")
    for param_tensor in Encrypted_model.state_dict():
        data = Encrypted_model.state_dict()[param_tensor].numpy()
        data = data.flatten()
        df = pd.DataFrame(data)
        df.to_csv(
            f"{folder}/{param_tensor}_{Encrypted_model.state_dict()[param_tensor].size()}.csv",
            index=False, header=False)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    filename = Path(args.model).name.title()
    main(use2d=True if "2d" in args.model else False, model_name=args.model, filters=int(filename.split("_")[2]))
