# Line-Level Handwriting Recognition in TensorFlow 2

This project contains code necessary to perform handwriting recognition
in TensorFlow 2. Using the provided scripts, the model can be trained and
also used for inference.

### Dependencies
* TensorFlow 2.x
* Python 3.x
* Numpy
* Pandas
* Pillow
* Matplotlib

Eventually, a .yaml file will be provided that will specify
all dependencies and that can be loaded into a Conda environment.

### Usage

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
* console (optional): Print inference results to the console and show images
* log_level (optional): TensorFlow log-level {0, 1, 2, 3} (default: 3)

