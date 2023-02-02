# Basic Instructions for Inference



## Install Conda

We use conda for dependency management.

https://docs.anaconda.com/anaconda/install/index.html

## Set up conda environment

There are two conda environments depending on the availability of a GPU on the system.

If you have a GPU to use during inference run the following command.

Otherwise, run the following command for installing the dependencies to run on the CPU.

## Download release data

Download the model weights from the github release.

## Edit inference_config.yaml

Edit inference config 

## Activate the conda environment

Run the following command to activate the conda environment.

```conda activate hwr_env```

## Run the model

Run the following command to run the model.

```python3 inference.py inference_config.yaml```

## View the results

The model will output results to the tsv path specified in the config file.

