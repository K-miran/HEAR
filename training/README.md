# Human Action Recognition via Neural Networks in the Clear


## Dependencies

We will need to install the followings:
- Python (>=3.6)
- PyTorch (>=1.3)
- torchvision
- OpenCV (>=3.4.1): it is needed for image processing.

In our experiment, we use CUDA (v10) and CUDNN (v8) for paralleization.
 
 
## Data Availability
We consider three datasets: J-HMDB, URFD, and Multicam. For each dataset, we used the [Deep High-Resolution network](https://arxiv.org/abs/1902.09212) for pose estimation, and modified the [nobos torch lib](https://github.com/noboevbo/nobos_torch_lib) for data transformation, normalization, and frame selection. 

The transformed samples from the three datasets are merged for analysis. The merged dataset is split randomly into training and testing sets that contain 70% (84 falls and 1346 non-falls) and 30% (29 falls and 579 non-falls), respectively. We modified the [Imbalanced Dataset Sampler](https://github.com/ufoym/imbalanced-dataset-sampler/) that can rebalance the class distributions and migigate overfitting over imbalanced datasets. 
The training and testing data can be found in the `TrainAndTestData` folder. The directory tree should look like this:
```
TrainAndTestData
├── inputs_test.pklt
├── inputs.pkl
├── label_test.pkl
└── label.pkl
```

## Examples

### Training
 The trained models can be found in the `training/saved_model` folder. Or you can train the model, and extract the training/testing data with the following commands:

```
python train.py
```

`train.py` accepts two optional arguments: 
- --filters: number of filters for the first convolutional layer (default: 128)
- --type: CNN network type (default: 2d)

For example, you can train by running the following command:
```
python train.py --filters 64 --type 1d
```  

### Testing
You can test with the trained model and get the accuracy in the cleartext. `test.py` requires one argument that is the path of the saved model file. For example, you can test the 2D-CNN-128 model by running the following command:
```
python test.py saved_model/2d_model_128_final.torch
```

You can extract the model parameters using `extract.py` by specifying the path of the model file.
```
python extract.py saved_model/2d_model_128_final.torch
```
It will extract the parameters to a csv file per each layer to `dataset` directory.

