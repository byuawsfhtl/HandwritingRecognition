# Line-Level Handwriting Recognition

This project contains code necessary to perform handwriting recognition
in TensorFlow 2. Using the provided scripts, the model can be trained and
also used for inference.

This project can be used by cloning the repository and running manually. However, it is also available in
[Anaconda Cloud](https://anaconda.org/BYU-Handwriting-Lab/hwr) and can be used in any Conda environment.

## Dependencies
* TensorFlow 2.x
* Python 3.x
* Numpy
* Pandas
* Matplotlib
* EditDistance

A .yaml file has been included that specifies the necessary dependencies. A conda environment can be
created and activated by running the following commands:

```
conda env create -f environment.yaml
conda activate hwr_env
```

### Conda Usage

Potentially, the easiest way to access the code is to import the [conda package](https://anaconda.org/byu-handwriting-lab/hwr)

```
conda install -c byu-handwriting-lab hwr
```

Code can then be accessed like any normal python package. For example, to use the recognition model,
you could write something like this:

```
from hwr.model import Recognizer
import hwr.dataset as ds

import tensorflow as tf
import numpy as np

model = Recognizer()

# Load some pretrained weights
model_weights_path = './some/path/to/model/weights'
model.load_weights(model_weights_path)

# The mapping between integers and characters
idx2char = ds.get_idx2char()

# Simulate creating an image with random numbers
fake_image = tf.constant(np.random.randn(1, 1024, 64, 1), dtype=tf.float32)

# Run the image through the recognition model
prob_dist = model(fake_image)
predictions = tf.argmax(prob_dist, axis=2)

# Convert to the character representation
str_predictions = ds.idxs_to_str_batch(predictions, idx2char)

print('Prediction:', str_predictions)
```

## Manual Usage

Using the actual codebase, you have access to the ```train.py``` and ```inference.py``` scripts.

### Training

Training can be run with the following command

```
python train.py <TRAIN_CONFIG_FILE>
```

Command Line Arguments:
* TRAIN_CONFIG_FILE (required): The path to the train configuration file. A train configuration file
  is provided as "train_config.yaml".

The train configuration file contains all the settings needed to run handwriting recognition. To run recognition on
your own model, simply modify the configuration file arguments. Explanations of the arguments are given below:

Configuration File Arguments:
* csv_path: The path to a tab-delimited CSV file containing | IMG_PATH | TRANSCRIPTION | 
  (Note that the IMG_PATH given in the CSV is relative to the location of the CSV)
* model_out: The path to store the trained model weights after training
* model_in: The path to pre-trained model weights to be loaded before training begins
* epochs: The number of epochs to train
* batch_size: The number of images in a mini-batch
* learning_rate: The learning rate the optimizer uses during training
* max_seq_size: The max number of characters in a line-level transcription
* img_size: The size which all images will be resized for training
* split_train: Whether or not to split the training set into a train/validation using the train_size parameter.
               Train = train_size, Val = (1 - train_size)
* train_size: The ratio used to determine the size of the train/validation split.
              Used ONLY if split_train is set to True
* show_graphs: Whether or not to show graphs of the loss after training
* metrics: Whether or not to include metrics other than loss on the validation set

Training Example:
* A few example images as well as a labels.csv file is included for reference in training and performing inference.
  This data is included in the data/example folder.
* Immediately after cloning this repository, you should be able to use the train.py script along with its
  train_config.yaml file to perform training. The model weights will be stored under
  data/model_weights/example_model/run1. You'll notice this as the *model_in* parameter in the inference_config file.
* Simply run the following commands:

`
python train.py train_config.yaml
`

### Inference

Inference can be run using the following command:

```
python inference.py <INFERENCE_CONFIG_FILE>
```

Command Line Arguments:
* INFERENCE_CONFIG_FILE (required): The path to the inference configuration file. An inference configuration
  file is provided as "inference_config.yaml".

Configuration File Arguments:
* img_path: The path to the images to be inferred
* out_path: The output path to the results of the inference
* model_in: The path to the pre-trained model weights to be used during inference
* img_size: The size which all images will be resized/padded for inference on the model
* batch_size: The batch size to be used when performing inference on the model (how many images inferred at once)


Inference Example:
* A few example test images are stored in the data/example/test_images folder for performing inference.
* After running the training script, the model weights will be stored according to the model_out parameter
  in the train_config file, data/model_weights/example_model/run1. In the example inference_config file, this
  path is already specified.
* After running the train.py script as specified above, you can perform inference on the example test images by
  running the following command:
  
`
python inference.py inference_config.yaml
`



## Build the Conda Package to be uploaded to Anaconda Cloud

Packaging python packages is done through the use of ```setup.py```  as well as ```meta.yaml```. Slight modifications
to these files may need to take place if dependencies to the code base change. The project can be packaged using the
following ```conda-build``` command.

```
conda-build ./conda.recipe -c defaults -c conda-forge
```

For the command to work, you may need to first activate the conda environment containing all of the project dependencies.

```
conda env create -f environment.yaml
conda activate hwr_env
```

Once the project has been packaged, the packaged file can be uploaded to Anaconda Cloud (Anaconda-Client is required
-- Linux-Only):

```
anaconda upload -u BYU-Handwriting-Lab <FILENAME>
```

