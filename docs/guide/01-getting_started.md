# Getting started

## Installation

You can install Recognite with pip:

```bash
pip install recognite
```

## Quickstart

The [Recognite GitHub repo](https://github.com/florisdf/recognite) contains an [example training script](https://github.com/florisdf/recognite/blob/main/examples/train.py) with which you can quickly start a recognition training. To use this script in your project run

```bash
# Clone the Recognite repo
git clone https://github.com/florisdf/recognite

# Copy the training script to your project
cp recognite/examples/train.py path/to/your/recognition_project

# Install the requirements of the training script
pip install -r recognite/examples/requirements.txt
```

The last line will install [Weights and Biases](https://wandb.ai), which is used for logging. Make sure to create an account and run `wandb login` from your command line.

Before you can start training, you'll need to define what data the model should train on. To do this, create a CSV file containing an `image` column and a `label` column. The `image` column contains image paths and the `label` column contains the corresponding labels. Save this as `data.csv` in the directory where you copied the `train.py` script to. Now you can run

```bash
python train.py --wandb_entity your_user_name --wandb_project your_project
```

Without any extra arguments, the script splits your dataset into a training (4/5) and validation (1/5) part and trains a ResNet-50 on the training subset by employing stochastic gradient descent (learning rate `0.01`, momentum `0.95`, weight decay `1e-5`, batch size `32`) to minimize the cross-entropy loss of the output of a dummy classification layer during `50` epochs. Of course, you can change all these hyperparameters. To get a list of the configurable parameters, run

```bash
python train.py --help
```

## Going further

The example training script will not suit everyone's needs. For example, you might be using a certain framework for your training loops, and prefer not to write them in plain PyTorch. That's fine! Recognite is designed with high modularity in mind, so you can easily pick and choose the pieces that you find useful.

In the following sections, we walk you through the main components of the library and show how you can use these in your own training scripts.
