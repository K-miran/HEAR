# HEAR: Secure Human Action Recognition by Encrypted Neural Network Inference 

## Introduction
Remote monitoring to support **aging in place** is an active area of research. Affordable webcams, together with cloud computing services (to run machine learning algorithms), can potentially bring significant social and health benefits. However, it has not been deployed in practice because of privacy concerns. In this work, we propose a secure service paradigm to reconcile the critical challenge by integrating machine learning techniques and fully homomorphic encryption. This is an official C++ implementation of [*Secure Human Action Recognition by Encrypted Neural Network Inference *](https://arxiv.org/abs/2104.09164) using the Residue Number System (RNS) variant of the [CKKS](https://eprint.iacr.org/2016/421.pdf) cryptosystem.

 
## Installation

We recommend to install `HEAR` into a C++ environment. 

### Dependencies 
- homebrew 
- texinfo 
- m4 >=1.4.16
- CMake >= 3.12
- Compiler: g++ version >= 6.0
- OpenMP 
  

### Data Availability
Our dataset contains two categories of data: 
1. Activities of Daily Living (ADL) were selected from the J-HMDB dataset. The selected action classes are clap, jump, pick, pour, run, sit, stand, walk, and wave.
2. The fall action class was created by the UR Fall Detection dataset (URFD) and the Multiple cameras fall dataset (Multicam). 

The three datasets are merged for analysis, and the merged dataset is split randomly into training and testing sets that contain 70% (84 falls and 1346 non-falls) and 30% (29 falls and 579 non-falls), respectively. To detect keypoint locations, we use the [Deep High-Resolution network](https://arxiv.org/abs/1902.09212) pre-trained with the [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/). 

For simplicity, the test samples (test input and labels) and pretrained models can be found in the `dataset` folder. 
- the test input: `test_input.csv`
- the test label: `test_label.csv`
- model parameters: The detailed model parameter numbers of convolutional neural networks (CNN) are found in each parameter folder. We follow the design rule of [ResNet](https://arxiv.org/abs/1512.03385) such that if the feature map size is halved, the number of filters in the convolutional layers is increased to doubled. In our experiments, we study one small net and one large net: CNN-64 and CNN-128, where 64 and 128 represent the number of filters in the first convolutional layer, respectively. For instance, `parameters_conv1d_64` contains pretrained parameters of 1d-CNN with 64 filters at the first layer. 

Your directory tree should look like this:
```
dataset
├── test_input.csv
├── test_label.csv
├── parameters_conv1d_64
├── parameters_conv1d_128
├── parameters_conv2d_64
└── parameters_conv2d_128
```

Alternatively, you can train the model and use it for secure inference. We refer to `training` folder for the details. 

### To Install HEAR 

1. Download the library or clond the repository with the following command: 
    ```
    git clone https://github.com/K-miran/HEAR.git
    ```
    We will call the directory that you cloned as ${HEAR_ROOT}.
2. Install the modified SEAL library (SEAL 3.4.5) with the following commands: 
    ```
    cd ${HEAR_ROOT}/testing/external/seal/native/src
    cmake. 
    make 
    ```
3. Install the HEAR library by running the following commands :
    ```
    cd ${HEAR_ROOT}/testing
    cmake . 
    make
    ```
It takes around 1 minute and 55 seconds to install the SEAL library and the HEAR library on a normal desktop computer (MacBook at 2.6GHz), respectively. 


## Examples

### Example Run
The following list of command-line arguments is required right after the name of the test program (`test`):
- HE-based CNN evaluation algorithm (e.g., hear or fhear): `hear` indicates the ordinary homomorhpic convolution evaluation method; `fhear` indicates the fast homomorphic convolution evaluation method. 
- The dimension of convolutions (e.g., 1 or 2): `1` and  `2` indicate the 1d and 2d convolution operations, respectively
- Number of threads
- The starting id number (<= 608)
- The ending id number (>=0)
- The number of channels at the first convolutional layer (e.g., 64 or 128)
- The evaluation method of the first convolutional layer (e.g., full or baby)
- The evaluation method of the second convolutional layer (e.g., full, baby, or giant)
- The evaluation method of the third convolutional layer (e.g., full, baby, or giant)

For instance, init `result` (predicted probabilities output directory) and run the test program with different inputs:
```
cd ${HEAR_ROOT}/testing/bin
mkdir result 
./test fhear 2 16 0 608 128 baby giant giant 
```
After running the program, the predicted results are stored in the `${HEAR_ROOT}/testing/bin/result` directory (e.g., `test_prob_conv2d_128_fhear_giant.txt` ). The actual predicted results from unencrypted computation can be found in the `dataset` folder (e.g., `test_prob_conv2d_128.txt` ), so you can compare these results with the actual predicted results. 

For convenience, you can check the classification performance (accuracy, sensitivity, specificity, precision, and F1-score) of predicted restuls from secure inference. 
The following list of command-line arguments is required right after the name of the test program (`accuracy`):
- HE-based CNN evaluation algorithm
- The dimension of convolutions 
- The number of channels at the first convolutional layer
- The evaluation method of the second convolutional layer

For instance, run the test program with different inputs:
```
cd ${HEAR_ROOT}/testing/bin 
./accuracy fhear 2 128 giant 
```
Then you will get the following output: 
```
(acc) HE vs true: acc=0.879934, sens=0.862069, spec=0.991364, prec=0.833333, F1score=0.847458
(acc) plain vs true: acc=0.881579, sens=0.862069, spec=0.991364, prec=0.833333, F1score=0.847458
```

### Example Output

We run the test program with the above different inputs on a machine equipped with an Intel Xeon Platinum 8268 at 2.9 GHz. Then you will get the following output: 

```
id=(0~608), #(threads)=16, #(channels)=(128,256,512)
Method=fhear, Mode=(baby,giant,giant)
> Key Generation: [16.0282 s] w/ 2.57677(GB)
> Prepare Network: (done) [2.02363 s] w/ 10.9518(GB)
> Encryption (608) [1.54149 s] w/ 12.9906(GB)
=========(0)=========
> Evaluation (B1): conv... act... avg... [0.561912 s] w/ 13.298(GB)
> Evaluation (B2): conv... act... avg... [2.05697 s] w/ 13.8576(GB)
> Evaluation (B3): conv... act... avg... [3.21544 s] w/ 14.0207(GB)
> Dense... [0.106654 s] w/ 14.1435(GB)
>> Total Eval Time = [3.3221 s] 
>  Decryption... [0.001464 s] w/ 14.1435(GB)
-0.0459498,-0.694072,-3.35113,11.7441,-5.94305,0.599028,1.04433,-1.70181,-4.10122,0.382085
=========(1)=========
> Evaluation (B1): conv... act... avg... [0.414887 s] w/ 14.1953(GB)
> Evaluation (B2): conv... act... avg... [2.32287 s] w/ 14.226(GB)
> Evaluation (B3): conv... act... avg... [3.10668 s] w/ 14.226(GB)
> Dense... [0.089413 s] w/ 14.226(GB)
>> Total Eval Time = [3.19609 s] 
>  Decryption... [0.001283 s] w/ 14.226(GB)
-3.47292,-1.2928,0.54706,-0.0126824,-2.98238,2.23631,-6.93829,1.87321,12.2315,-4.00186
...
=========(607)=========
> Evaluation (B1): conv... act... avg... [0.393676 s] w/ 16.5045(GB)
> Evaluation (B2): conv... act... avg... [1.52436 s] w/ 16.5069(GB)
> Evaluation (B3): conv... act... avg... [2.28856 s] w/ 16.5069(GB)
> Dense... [0.067146 s] w/ 16.5069(GB)
>> Total Eval Time = [2.3557 s] 
>  Decryption... [0.001262 s] w/ 16.5069(GB)
-5.15543,3.26147,-4.14035,14.0757,-3.35955,-5.83737,2.14,-2.43445,-0.215976,-0.345049
```
