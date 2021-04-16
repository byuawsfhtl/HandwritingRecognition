# Line-Level Handwriting Recognition

This project contains code necessary to perform handwriting recognition
in TensorFlow 2. Using the provided scripts, the model can be trained and
also used for inference.

This project can be used by cloning the repository and running manually. However, it is also available in
[Anaconda Cloud](https://anaconda.org/BYU-Handwriting-Lab/hwr) and can be used in any Conda environment.

## Dependencies
* Python 3.x
* TensorFlow 2.x
* Numpy
* Pyyaml
* Pandas
* Matplotlib
* Tqdm
* EditDistance

A .yaml file for each supported platform has been included that specifies the necessary dependencies. A
conda environment for MacOS/Windows/Linux can be created and activated by running the following commands:

```
conda env create -f environment_linux.yaml  # or environment_macos.yaml, environment_windows.yaml
conda activate hwr_env
```

## Usage With Provided Scripts

Using the code available in this repository, you have access to the ```train.py```, ```inference.py```,
and ```test.py``` scripts.

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
* train_csv_path: The path to a tab-delimited CSV file containing training data information formatted as:
                  | IMG_PATH | Transcription |
* val_csv_path: The path to a tab_delimited CSV file containing validation data information formatted as:
                  | IMG_PATH | Transcription |     This field may be left blank if Split Train is set to True.
* split_train_size: The ratio used to determine the size of the train/validation split. If split_train_size is set
                    to 0.8, then the training set will contain 80% of the data, and validation 20%. The dataset is
                    not shuffled before being split. If a val_csv_path is given, this parameter will not be used.
                    Otherwise, the training set will be split using this parameter.
* apply_noise_augmentation: Whether or not to apply the noise augmentation to the training dataset
* apply_bleedthrough_augmentation: Whether or not to apply the bleedthrough augmentation to the training dataset
* apply_grid_warp_augmentation: Whether or not to apply the grid warp augmentation to the training dataset
* grid_warp_interval: The interval in pixels between control points in the grid warp augmentation
* grid_warp_stddev: The standard deviation required in the grid warp augmentation                   
* model_out: The path to store the trained model weights after training
* model_in: The path to pre-trained model weights to be loaded before training begins
* epochs: The number of epochs to train
* batch_size: The number of images in a mini-batch
* learning_rate: The learning rate the optimizer uses during training
* max_seq_size: The max number of characters in a line-level transcription
* img_size: The size which all images will be resized for training
* charset: String including all characters to be represented in the network (abcdef1234...)
           If no characters are specified, the default is used.

Training Example:
* A few example images as well as a labels.csv file is included for reference in training and performing inference.
  This data is included in the data/example folder.
* Immediately after cloning this repository, you should be able to use the train.py script along with its
  train_config.yaml file to perform training. The model weights will be stored under
  data/model_weights/example_model/run1. You'll notice this as the *model_in* parameter in the inference_config file.
* Please not that this example data is meant to show usage for the system. It is by no means sufficient to actually
  train a recognition model.
* To use the example parameters, simply run the following command with the original ```train_config.yaml``` file:

    ```
    python train.py train_config.yaml
    ```

### Inference

Inference can be run using the following command:

```
python inference.py <INFERENCE_CONFIG_FILE>
```

The purpose of the ```inference.py``` script is to perform inference on a directory of text line images. The output
of the ```inference.py``` script will be a csv containing | IMG_PATH | PREDICTED_TRANSCRIPTION | 

Command Line Arguments:
* INFERENCE_CONFIG_FILE (required): The path to the inference configuration file. An inference configuration
  file is provided as "inference_config.yaml".

Configuration File Arguments:
* img_path: The path to the images to be inferred
* img_path_subdirs: Boolean whether or not to include subdirectories of img_path
* out_path: The output path to the results of the inference
* model_in: The path to the pre-trained model weights to be used during inference
* img_size: The size which all images will be resized/padded for inference on the model
* batch_size: The batch size to be used when performing inference on the model (how many images inferred at once)
* charset: String including all characters to be represented in the network (abcdef1234...)
           If no characters are specified, the default is used.
* use_wbs: Boolean indicating whether or not word beam search should be used for decoding. If false, best-path is used.
* wbs_beam_width: The beam width needed for the word beam search algorithm
* wbs_word_charset: String containing all characters observed in words (non-word_charset)
* wbs_dictionary_path: A path to a file containing a list of words that the wbs should constrain to. If none given wbs will use the default dictionary


Inference Example:
* A few example test images are stored in the data/example/test_images folder for performing inference.
* After running the training script, the model weights will be stored according to the model_out parameter
  in the train_config file, data/model_weights/example_model/run1. In the example inference_config file, this
  path is already specified.
* After running the train.py script as specified above, you can perform inference on the example test images by
  running the following command:
  
    ```
    python inference.py inference_config.yaml
    ```
  
### Test

Model testing can be done using the following command:

```
python test.py <TEST_CONFIG_FILE>
``` 

The purpose of the ```test.py``` script is to get an idea of how a trained model is
performing. By running ```test.py```, you can receive Character/Word Error Rates for
Best-Path/Word-Beam-Search decoding algorithms as well as prediction examples for
qualitative analysis.

Command Line Arguments:
* TEST_CONFIG_FILE: The path to the test configuration file. A test configuration file
  is provided as "test_config.yaml".

Command Line Arguments:
* csv_path: The path to the csv file
* dataset_eval_size: How much of the dataset should be used when testing and acquiring error rates. Float between 0-1.
* batch_size: The number of images to be used in a batch
* max_seq_size: The max number of characters in a line-level transcription
* charset: String including all characters to be represented in the network (abcdef1234...)
           If no characters are specified, the default is used.
* show_predictions: Boolean indicating whether or not to print the bp/wbs predictions along with label
* use_wbs: Boolean indicating whether or not word beam search should be used for decoding. If false, best-path is used.
* wbs_beam_width: The beam width needed for the word beam search algorithm
* wbs_word_charset: String containing all characters observed in words (non-word_charset)
* wbs_dictionary_path: A path to a file containing a list of words that the wbs should constrain to. If none given wbs will use the default dictionary

Test Example:
* We can use the same images/labels.csv files as are used for training to test the model. This data is included in the
  data/example folder
* After running the training script, the model weights will be stored according to the model_out parameter
  in the train_config file, data/model_weights/example_model/run1. We can use the weights of this trained model to
  run our test script. In the example test_config file, this path is already specified.
* To run this example you can use the provided config file:

    ```
    python test.py test_config.yaml
    ```

## Usage With Conda Package

Potentially, the easiest way to access the code is to import the [conda package](https://anaconda.org/byu-handwriting-lab/hwr)
that is available on Anaconda-Cloud. No cloning of this repository is necessary.

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

