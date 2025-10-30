# ANIDSC
 Adversarial NIDS Chain

# Setting up directories
It is recommended to have two folders, one for dataset and one for actual code.

The dataset folder structure should be like the following: ./datasets/{dataset_name}/pcap/{filename}

There is no restriction on code structure 

# Setting Up environment
install docker

run
`docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v "path/to/this/folder":/workspace/ANIDSC -v "path/to/datasets":/workspace/datasets kihy/nids_framework`

if gpu is not set up you can remove --gpus all 

go to code folder 
`cd /workspace/ANIDSC`
and add code there.

The ANIDSC package should already be installed, just need to import.

# Running the code
The examples folder include example scripts to run this package (Note these scripts are examples used for actual dataset):
* adv_gen.py shows how to conduct adversarial attack, still under development
* live_detection.py shows how to create NIDS pipeline 
* summarize_results.py contains functions to plot the anomaly scores
* run_lager_xai.py runs all configurations in the ./configs/ directory, and applies XAI methods to LAGER

You can make your own scripts to customize behavior and functionality.

Place the dataset (raw .pcap files) into `./ANIDSC/datasets/pcap`. You must run `uq_feature_extraction()` or similar customized function to create feature extraction components for the pipeline. These components will be saved to `./ANIDSC/datasets/[chosen_feature_extractor]/feature_extractors`. 

Note, that you must define your own feature extraction code in order to choose how the features get extracted, otherwise it will default to "AfterImage". 

All evaluations will be saved under `./ANIDSC/datasets/[chosen_feature_extractor]/results`. 

There is also feature folder for scenario testing that can be used as example. This is referred method to run small test data

To run some test data, cd into the features directory and run 

`behave 1_feature_extraction_chain.feature
behave 2_basic_chain.feature
behave 1_graph_chain.feature`

# extending the code
To add a new compenents, extend classes described in under src/ANIDSC/base_files 
