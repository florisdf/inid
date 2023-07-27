# Recognite


[![](https://img.shields.io/pypi/v/recognite.svg)](https://pypi.org/project/recognite/)
[![](https://readthedocs.org/projects/recognite/badge/?version=latest)](https://recognite.readthedocs.io/)


Recognite is a library to kickstart your next PyTorch-based recognition project. Some interesting features include:

- You can choose from nearly **80 different base models** for your recognition model: classics like AlexNet, GoogLeNet, VGG, Inception, ResNet, but also more recent models like ResNeXt, EfficientNet, and transformer-based models like ViT and SwinTransformer.
- You can easily evaluate your model model directly for a **recognition task**, where *query* samples are compared with a *gallery*, and none of the samples have a class that was used during training.
- By changing only a single argument, you can **cross-validate sets of hyperparameters** without much effort.


## Installation

You can install Recognite with pip:

```bash
pip install recognite
```

## Quickstart

This repo contains a [basic training script](examples/basic/train.py) with which you can quickly start a recognition training. To use this script in your project, you can clone the repository, copy the script into your project directory and install the script's requirements:

```bash
# Clone the Recognite repo
git clone https://github.com/florisdf/recognite

# Copy the training script to your project
cp recognite/examples/basic/train.py path/to/your/recognition_project

# Install the requirements of the training script
pip install -r recognite/examples/basic/requirements.txt
```

> The last line installs [Weights and Biases](https://wandb.ai), which is used for logging. Make sure to create an account and run `wandb login` from your command line.

The training script trains a recognition model of your choice on a dataset you define, using tools from the Recognite library. The dataset should be given as a CSV file (`--data_csv`) with two columns: `image` (containing image paths) and `label` (containing the corresponding labels). We split the unique labels of the dataset into 5 folds. Labels in the fold defined by `--val_fold` are used for validation. The others are used for training. During validation, we measure the model's top-1 accuracy when classifying a set of queries by comparing the query embeddings with the embeddings of a set of reference samples (`--num_refs` per validation label). This accuracy is logged to Weights and Biases (see `--wandb_entity` and `--wandb_project`).

Each image is uniformly resized such that its shortest side has a fixed size (`--size`). For training images, we then take a square crop of that size at a random location in the image. For the validation images, we crop out the square center of the image.

For the model, you can choose from a large number of pretrained classifiers, see `--model_name` and `--model_weights`. The model's final fully-connected layer is adjusted to the number of classes in the training set and is then trained for `--num_epoch` epochs by optimizing the softmax cross-entropy loss with stochastic gradient descent, configured by `--batch_size`, `--lr`, `--momentum` and `--weight_decay`.

For example, with the following command, we train a ResNet-18 model with [default pretrained weights](https://pytorch.org/vision/main/models.html) for 30 epochs on images from `data.csv` using a learning rate of `0.01`, a momentum of `0.9`, and a weight decay of `1e-5`. As validation set, we use the labels of the first fold (index `0`) and we use `1` reference sample per label in the gallery set.


```bash
python train.py \
    --model_name=resnet18 --model_weights=DEFAULT \
    --data_csv=data.csv --val_fold=0 --num_refs=1 --size=224 \
    --num_epochs=30 --lr=0.01 --momentum=0.9 --weight_decay=1e-5 \
    --wandb_entity=your_user_name --wandb_project=your_project
```

For more details on the different command line arguments, you can run

```bash
python train.py --help
```

## More information

See [the docs](https://recognite.readthedocs.io/) for more information and examples with Recognite.
