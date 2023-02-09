# Basic Instructions for Inference

Run all commands from the root directoy of this repository.

## Install Conda

We use conda for dependency management.

[Anaconda Install](https://docs.anaconda.com/anaconda/install/index.html)

## Set up conda environment

There are two conda environments depending on the availability of a GPU on the system.

If you have a GPU to use during inference run the following command.

```conda env create -f environment_gpu.yaml```

Otherwise, run the following command for installing the dependencies to run on the CPU.

```conda env create -f environment_cpu.yaml```

## Download release data

Download the model weights and sample data from the github release. These files are contained in sample.zip.

Extract the files and place the sample folder in the data folder of this directory. The data and inference_config.yaml can be placed anywhere, but these instructions assume that it will be placed there.

## Edit inference_config.yaml

Edit inference config to specify input directory and output file path. This is already done for the sample images, no changes are needed unless using images other than the sample.

## Activate the conda environment

Run the following command to activate the conda environment.

```conda activate hwr_env```

## Run the model

Run the following command to run the model.

```python3 inference.py data/sample/inference_config.yaml```

## View the results

The model will output results to the tsv path specified in the config file.

By default, the output file will be located at:

```data/sample/inference.csv```

