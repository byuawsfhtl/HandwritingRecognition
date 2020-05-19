# Line-Level Handwriting Recognition in TensorFlow 2

This project contains code necessary to perform handwriting recognition
in TensorFlow 2. Currently, training is not available and will only
perform inference with pre-trained weights.

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

Inference can be run with the following command

`python inference.py --img_path ./data/test_images --out_path ./`

Optionally, inference can be printed to the console instead
of an output file. A plot with the image will also be shown.
To continue on to the next image, simply close the image window.

`python inference.py --img_path ./data/test_images --console`
