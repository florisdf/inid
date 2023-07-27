# Getting started

## Installation

You can install Recognite with pip:

```bash
pip install recognite
```

## Quickstart

The [Recognite GitHub repo](https://github.com/florisdf/recognite) contains a [basic training script](https://github.com/florisdf/recognite/blob/main/examples/basic/train.py) with which you can quickly start a recognition training. To use this script in your project, you can clone the repository, copy the script into your project directory and install the script's requirements:

```bash
# Clone the Recognite repo
git clone https://github.com/florisdf/recognite

# Copy the training script to your project
cp recognite/examples/basic/train.py path/to/your/recognition_project

# Install the requirements of the training script
pip install -r recognite/examples/basic/requirements.txt
```

```{note}
The last line installs [Weights and Biases](https://wandb.ai), which is used for logging. Make sure to create an account and run `wandb login` from your command line.
```

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

For more information on the different command line arguments, you can run

```bash
python train.py --help
```

## Going beyond the basics

While the basic example training script already contains many useful functionalities, Recognite can be used to add other interesting features to your training script, such as

- More insightful [metrics](./04-evaluation): the distribution of hard negative and hard positive scores, the distribution of Average Precisions, the mAP, the distribution of thresholds at maximum F1 score,...
- [Averaging of reference embeddings](./05-utils.md#avg-ref-embs)
- [Running maximum and minimum of metrics](./05-utils.md#running-extr)
- [Three-croping during validation](./05-utils.md#three-crop)

In the [advanced training script](https://github.com/florisdf/recognite/blob/main/examples/advanced/train.py), we have added these features, along with some extras such as:

- Configurable [random resized cropping](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)
- Learning rate warm-up
- Only saving the checkpoint when the model is the best yet (according to a chosen metric)
- Freezable model parameters
- Configurable number of folds

To use the advanced training script, run

```bash
# Copy the training script to your project
cp -a recognite/examples/advanced/. path/to/your/recognition_project

# Install the requirements of the training script
pip install -r recognite/examples/advanced/requirements.txt
```

Check out the script's `--help` for more information.

## Using Recognite in your own set-up

The example training scripts will not suit everyone's needs. For example, you might be using a certain framework for your training loops, and prefer not to write them in plain PyTorch. That's fine! Recognite is designed with high modularity in mind, so you can easily pick and choose the pieces that you find useful.

In the following sections, we walk you through the main components of the library and show how you can use these in your own training scripts.
