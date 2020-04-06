# Deep Packet

Details in blog post: https://blog.munhou.com/2020/04/06/Pytorch-Implementation-of-Deep-Packet-A-Novel-Approach-For-Encrypted-Tra%EF%AC%83c-Classi%EF%AC%81cation-Using-Deep-Learning/

## How to Use

* Clone the project
* Download the train and test set I created at [here](https://drive.google.com/file/d/1ID-iFRnEOahH4zsDU7PmdAv4_833lLZo/view?usp=sharing), or download the [full dataset](https://www.unb.ca/cic/datasets/vpn.html) if you want to process the data from scratch.
* Run python codes with the docker image:
```bash
docker run -it \
-v /path/to/the/code:/data \
mhwong2007/deep_packet \
bash
```
* If you want to run Jupyter notebook, use the folowwing command:
```bash
docker run -it \
-v /path/to/the/code:/data \
-p 8888:8888 \
mhwong2007/deep_packet \
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --allow-root
```
* If you want to build the environment yourself, please install the dependencies and libraries in the [Dockerfile](Dockerfile)

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

Download the pre-trained models [here](https://drive.google.com/file/d/1f0zjzuerhbWmXrgob7uDGQt-fdZrJ50V/view?usp=sharing).