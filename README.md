# Simple Captcha Detector

## Requirements

We need to use Python 3 in order to run this package. Run:
```
pip install -r requirements.txt
```
to install all requirements.

To train the model, download and unpack the archive to the repository root directory by
```
bash download.sh
```

## Training the Model

To train the model, simply run
```
python train.py input_directory_path output_directory_path
```
which will save the model to `model.pt`. To change this file name, specify `--output-model filename` parameter.

The training takes approximately 15 minutes on a MacBook with 2.6 GHz 6-core Intel Core i7 and 16 GB memory.

## Running the Model

In a Python program, import `Captcha` class from `captcha.py` and run the program by the following example:
```
from captcha import Captcha
...
c = Captcha()  # customise the model path by `model_path` argument; set verbose to True to output scores
c("input/input100.jpg", "output/output100.txt")
```

## Model Description

The model specified in `model.py` is a simple deep learning model with a convolutional neural network (CNN),
which slides through the width (horizontal) dimension of the image to extract characters.
Assuming the image has the same dimension, we treat the height pixels as the input channel.
The colour information is discarded as what matters is the shade.

On top of it, we apply layer normalization for better generalization and dropout
(only invoked during training) to prevent overfitting. Then, we apply 5 attention probability distributions
with a single query each and sinusoidal position encoding (Vaswani et al., NIPS 2017) as the attention key.
This is to learn the positions each of the 5 Captcha characters. The 5 attention distributions are applied to
the CNN output resulting in 5 hidden vectors, corresponding to the 5 characters.

Then, they are fed to an output projection layer with 36 elements, corresponding to 10 numerical digits
and 26 alphabetical characters, where the element with the maximum probability score is the output.

Training is done by Adam algorithm, a stochastic gradient descent (SGD) with momentum. We use the Torch
implementation of AdamW, which adds L2 regularization (weight decay) to prevent overfitting.
