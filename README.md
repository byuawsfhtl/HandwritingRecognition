# Line-Level Handwriting Recognition in TensorFlow 2

This project contains code necessary to perform handwriting recognition
in TensorFlow 2. Using the provided scripts, the model can be trained and
also used for inference.

## Dependencies
* TensorFlow 2.x
* Python 3.x
* Numpy
* Pandas
* Pillow
* Matplotlib

A .yaml files has been included that specifies the necessary dependencies. A conda environment can be
created and activated by running the following commands:
`
conda env create -f environment.yaml
conda activate hwr_env
`

## Usage

This project can be used by cloning and repository and running manually. However, it is also available in
[Anaconda Cloud](https://anaconda.org/BYU-Handwriting-Lab/hwr) and can be used in any Conda environment.

### Conda Usage

Coming soon!

### Manual Usage

Training can be run with the following command

`python train.py --img_path ./data/training/images --label_path ./data/training/labels.csv`

Optionally, a number of command line arguments can be used to alter training behavior.
The full list of parameters include:
* img_path (required): The path to the images in the dataset
* label_path (required): The path to the label CSV (Format: Word | Transcription - Tab-Delimited, No-Header)
* show_graphs (optional): Whether or not to show graphs of metrics after training (default: don't show graphs)
* log-level (optional): TensorFlow log-level {0, 1, 2, 3} (default: 3)
* model_out (optional): The path to store the model weights (default: ./data/model_weights/hwr_model/run1)
* epochs (optional): The number of epochs to train (default: 100)
* batch_size (optional): The number of images in a mini-batch (default: 100)
* learning_rate (optional): The learning rate the optimizer uses during training (default: 4e-4)
* max_seq_size (optional): The max number of characters in a line-level transcription (default: 128)
* train_size (optional): The ratio used to determine the size of the train/validation sets (default: 0.8)
* tfrecord_out (optional): The path to the created tfrecords file (default: './data/misc/data.tfrecords)
* weights_path (optional): The path to pre-trained model weights (default: None)

Inference can be run with the following command

`python inference.py --img_path ./data/test_images --out_path ./`

Optionally, inference can be printed to the console instead
of an output file. A plot with the image will also be shown.
To continue on to the next image, simply close the image window.

`python inference.py --img_path ./data/test_images --console`

The full list of inference arguments include:
* img_path (required): The path to images to be inferred
* out_path (required if console not specified): The output path to the results of the inference
* weights_path (required: The path to the pre-trained model weights
* console (optional): Print inference results to the console and show images
* log_level (optional): TensorFlow log-level {0, 1, 2, 3} (default: 3)


### Build the Conda Package to be uploaded to Anaconda Cloud

Packaging python packages is done through the use of ```setup.py```  as well as ```meta.yaml```. Slight modifications
to these files may need to take place if dependencies to the code base change. The project can be packaged using the
following ```conda-build``` command.

`
conda-build ./ -c defaults -c conda-forge
`

For the command to work, you may need to first activate the conda environment containing all of the project dependencies.

`
conda env create -f environment.yaml
conda activate hwr_env
`

Once the project has been packaged, the packaged file can be uploaded to Anaconda Cloud (Anaconda-Client is required):

`
anaconda upload -u BYU-Handwriting-Lab <FILENAME>
`

