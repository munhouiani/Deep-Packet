# Deep Packet

Details in blog
post: https://blog.munhou.com/2020/04/05/Pytorch-Implementation-of-Deep-Packet-A-Novel-Approach-For-Encrypted-Tra%EF%AC%83c-Classi%EF%AC%81cation-Using-Deep-Learning/

## EDIT: 2022-09-27

* Update dataset and model
* Update dependencies
* Add more data to `chat`, `file_transfer`, `voip`, `streaming` and `vpn_voip`
* Remove tor and torrent related data as they are no longer available

## EDIT: 2022-01-18

* Update dataset and model

## EDIT: 2022-01-17

* Update code and model
* Drop `petastorm`, use huggingface's `datasets` instead for data loader

## How to Use

* Clone the project
* Create environment via conda
    * For Mac
      ```bash
      conda env create -f env_mac.yaml
      ```
    * For Linux (CPU only)
      ```bash
      conda env create -f env_linux_cpu.yaml
      ```
    * For Linux (CUDA 10.2)
      ```bash
      conda env create -f env_linux_cuda102.yaml
      ```
    * For Linux (CUDA 11.3)
      ```bash
      conda env create -f env_linux_cuda113.yaml
      ```
* Download the train and test set I created
  at [here](https://drive.google.com/file/d/1EF2MYyxMOWppCUXlte8lopkytMyiuQu_/view?usp=sharing), or download
  the [full dataset](https://www.unb.ca/cic/datasets/vpn.html) if you want to process the data from scratch.

## Data Pre-processing

```bash
python preprocessing.py -s /path/to/CompletePcap/ -t processed_data
```

## Create Train and Test

```bash
python create_train_test_set.py -s processed_data -t train_test_data
```

## Train Model

Application Classification

```bash
python train_cnn.py -d train_test_data/application_classification/train.parquet -m model/application_classification.cnn.model -t app
```

Traffic Classification

```bash
python train_cnn.py -d train_test_data/traffic_classification/train.parquet -m model/traffic_classification.cnn.model -t traffic
```

## Evaluation Result

### Application Classification

![](https://blog.munhou.com/images/deep-packet/cnn_app_classification.png)

### Traffic Classification

![](https://blog.munhou.com/images/deep-packet/cnn_traffic_classification.png)

## Model Files

Download the pre-trained
models [here](https://drive.google.com/file/d/1LFrx2us11cNqIDm_yWcfMES5ypvAgpmC/view?usp=sharing).

## Elapsed Time

### Preprocessing

Code ran on AWS `c5.4xlarge`

```
7:01:32 elapsed
```

### Train and Test Creation

Code ran on AWS `c5.4xlarge`

```
2:55:46 elapsed
```

### Traffic Classification Model Training

Code ran on AWS `g5.xlarge`

```
24:41 elapsed
```

### Application Classification Model Training

Code ran on AWS `g5.xlarge`

```
7:55 elapsed
```
