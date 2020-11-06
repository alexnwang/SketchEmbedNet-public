# SketchEmbedNet-public

This repository contains code for the research paper:  
**SketchEmbedNet: Learning Novel Concepts by Imitating Drawings**. Alexander Wang, Mengye Ren, Richard S. Zemel. NeurIPS 2019. [[arxiv](https://arxiv.org/abs/2009.04806)]

## Dependencies
* python 3.7+
* tensorflow 2.1+
* svgwrite
* cairosvg
* rdp
* svgpathtools

## Setup
Clone the repository and prepare the following datasets. QuickDraw must be downloaded as ```.npz``` files and processed locally.
The rest of the datasets have been zipped and can be used directly. 

Training data:
* QuickDraw [[link](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn;tab=objects?pli=1)] [[gcloud](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false)]
* Sketchy [[download](https://drive.google.com/file/d/1Kc9F98ShwCSOcNG35wngByiDiZDl9MGd/view?usp=sharing)]

Test data:
* Omniglot [[download](https://drive.google.com/file/d/1t1ejr7WhdzlV0A1Nsg2yAVE5h4MtmzQI/view?usp=sharing)] 
* miniImageNet [[download](https://drive.google.com/file/d/18F14cLamAqTCtf-U7BILF0t0E0hHBYEl/view?usp=sharing)] 

The original, unprocessed datasets can be found here: [[Sketchy](https://sketchy.eye.gatech.edu/)] [[Omniglot](https://github.com/brendenlake/omniglot)] [[mini-ImageNet](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view)]

### Directories
Prepare a location and variable `$DIR` and `$DATA_DIR`. The former is where all experiment folders will be stored and the latter is where all data will be stored. 

### Data setup
**QuickDraw**  
Download all .npz files and place them in the directory   
```data_dir/quickdraw/raw/ ...Quickdraw .npz files...```
and run the python script
```
python prepare_data.py \
    --dir=$DIR \
    --data_dir=$DATA_DIR \
    --id=prepare_data \
    --logfile={LOG_NAME} \
    --dataset=quickdraw \
    --dataset_cfgset=quickdraw
    --dataset_cfgs="split={SPLIT_NAME}" \
```

**Sketchy**  
Run the following command to setup the Sketchy data:  
```
mkdir data_dir/sketchy/caches
tar -xf sketchy_cache.tar.gz -C data_dir/sketchy/caches
```

**Omniglot**  
Run the following command to setup the Omniglot data:  
```
mkdir data_dir/fs_omniglot/caches
tar -xf omniglot_cache.tar.gz -C data_dir/fs_omniglot
```  
The cache of the Omniglot dataset is only for the Vinyals split reported in our main experiments which uses rotations or characters as new classes.
To use the lake split the dataset must be re-processed from the raw data.

**mini-ImageNet**  
Run the following command to setup the mini-ImageNet data:  
```
mkdir data_dir/miniimagenet/caches
tar -xf miniimagenet_cache.tar.gz -C data_dir/miniimagenet
```

**Note**  
The Sketchy and Quickdraw datasets follow different paradigms from the Omniglot and mini-ImageNet datasets. The former have different
splits that contain different sets of classes that are named within the quickdraw/caches folder, where each one is a split. The latter
2 datasets contain a folder for each class in the dataset.

### Preprocessing Data
Data can be processed using the ```prepare_data.py``` script. `fs_omniglot` and `miniimagenet` do not require a `class_list` configuration,
however, `quickdraw` and `sketchy` do.
```
python prepare_data.py \
    --id=prepare_data \
    --logfile=prepare_fs_omniglot \
    --dataset={DATASET} \
    --dataset_cfgset={DATASET_CONFIG_SET} \
    --dataset_cfgs={DATASET_CONFIG_OVERRIDE}
```
* `DATASET` values can be `quickdraw` `sketchy` `fs_omniglot` `miniimagenet`
* `DATASET_CONFIG_SET` are all decorator names in `/configs/{DATASET}_configs`
* `DATASET_CONFIG_OVERRIDE` is a string of format `"config1=value1,config2=value2..."`
    * For the `Quickdraw` and `Sketchy` datasets, a `split={SPLIT_NAME}` override is required where `{SPLIT_NAME}` will be a unique identifier 
    for the data with a the current set of classes being included.

## General Operation
Experiments are run through python files that follow the `run_{}.py` naming convention. All of these files accept common arguments

* `--dir` Directory of all experiments, where single experiment directories are kept.
* `--data_dir` Directory containing all datasets, same as `data_dir` in the setup section.
* `--check_numerics` Debugging flag, enables the tensorflow to check for `NaN` and `inf` values.
* `--id` unique identifier with a corresponding folder where saved models and logs are stored.
* `--logfile` the name of the log file for the given experiment stored in the corresponding ID folder.

### Training the model
```
python run_experiment.py \
    --dir=$DIR \
    --data_dir=$DATA_DIR \
    --logfile={LOG_NAME} \
    --id={MODEL_ID} \
    --model={MODEL} \
    --model_cfgset={MODEL_CONFIG_SET} \
    --model_cfgs={MODEL_CONFIG_OVERRIDE} \
    --train_dataset={DATASET} \
    --train_dataset_cfgset={DATASET_CONFIG_SET} \
    --train_dataset_cfgs={DATASET_CONFIG_OVERRIDE} \
    --train_steps=300000 \
    --save_freq=75000
```
* `MODEL_ID` is anything and for referencing the trained model for later tasks.
* `MODEL` values can be `drawer_enc_block`, `drawer_enc_resnet12`, `classifier`, `vae_enc_block`
* `MODEL_CONFIG_SET` can be any names in `/configs/{MODEL}_configs`
* `MODEL_CONFIG_OVERRIDE` is a string of format `"config1=value1,config2=value2..."`
* `DATASET` values can be `quickdraw` `sketchy` `fs_omniglot` `miniimagenet`
* `DATASET_CONFIG_SET` are all decorator names in `/configs/{DATASET}_configs`
* `DATASET_CONFIG_OVERRIDE` is a string of format `"config1=value1,config2=value2..."`
    * For the `Quickdraw` and `Sketchy` datasets, a `split={SPLIT_NAME}` override is required.
* Evaluation can performed by including flags `eval_dataset, eval_dataset_cfgset, eval_dataset_cfgs, eval_freq`.

Training can be parallelized using Horovod by adding the flag `--distributed=True` and running  the script through the `horovodrun`.  

### Evaluating the model
```
python run_full_eval.py \
    --dir=$DIR \
    --data_dir=$DATA_DIR \
    --logfile={LOG_NAME} \
    --id={MODEL_ID} \
    --model={MODEL} \
    --model_cfgset={MODEL_CONFIG_SET} \
    --model_cfgs={MODEL_CONFIG_OVERRIDE} \
    --natural={NATURAL} \
    --sample={SAMPLE} \
    --gen={GEN} \
    --usfs={USFS} \
    --checkpoint={CHECKPOINT}
```
* `MODEL_ID` should match the ID assigned during training to the model to be evaluated.
* `MODEL` `MODEL_CONFIG_SET` and `MODEL_CONFIG_OVERRIDE` should match model training time.
* `SAMPLE` boolean to sample sketches from the model
* `GEN` boolean to perform one-shot generation classification
* `USFS` boolean to perform UnSupervised Few-Shot classification
* `CHECKPOINT` boolean to perform USFS over training checkpoints

**Generative Eval**  
To perform this evaluation, the following splits and classifier trainings must be prepared.


### Compositionality experiments
First download and extract [[this](https://drive.google.com/file/d/1Kc9F98ShwCSOcNG35wngByiDiZDl9MGd/view?usp=sharing)] additional datasets into the Quickdraw dataset folder.
It contains a number of splits that are used for our generated examples.
```
mkdir data_dir/quickdraw/caches
tar -xf compositionality_caches.tar.gz -C data_dir/quickdraw/caches
```

```
python run_compositionality_exp.py \
    --dir=$DIR \
    --data_dir=$DATA_DIR \
    --logfile={LOG_NAME} \
    --id={ID} \
    --drawer_id={DRAW_MODEL_ID} \
    --drawer_model={DRAW_MODEL} \
    --drawer_cfgset={DRAW_MODEL_CONFIG_SET} \
    --drawer_cfgs={DRAW_MODEL_CONFIG_OVERRIDE} \
    --vae_id={VAE_MODEL_ID} \
    --vae_model{VAE_MODEL} \
    --vae_cfgset={VAE_MODEL_CONFIGSET} \
    --vae_cfgs={VAE_MODEL_CONFIG_OVERRIDE}
```
* `ID` is where compositionality results will be stored
* `DRAW_MODEL_ID` ID assigned to drawing model durining training
* `DRAW_MODEL` `DRAW_MODEL_CONFIG_SET` and `DRAW_MODEL_CONFIG_OVERRIDE` should match draw model training time.
* `VAE_MODEL_ID` ID assigned to VAE model during training
* `VAE_MODEL` `VAE_MODEL_CONFIG_SET` and `VAE_MODEL_CONFIG_OVERRIDE` should match VAE model training time.

## Pretrained Models
We provide the model checkpoints for our most performant few-shot classification models trained on the Quickdraw and Sketchy datasets.
The archives can be directly extracted into an empty experiment folder `$DIR/{id}/checkpoints` and can then be accessed by the `--id` flag.

[[Quickdraw](https://drive.google.com/file/d/1b1LFFb3ZhTzQFN3ZBj0ANJblXtxmysgX/view?usp=sharing)] - Used for Omniglot task  
[[Sketchy](https://drive.google.com/file/d/1N-Bf2UJPOdl9oxEoAmMcmc9GUHDVcTgz/view?usp=sharing)] - Used for mini-ImageNet task
## Citing our work
If you use this code or our work, please consider citing us with the below citation:
```
@misc{wang2020sketchembednet,
      title={SketchEmbedNet: Learning Novel Concepts by Imitating Drawings}, 
      author={Alexander Wang and Mengye Ren and Richard Zemel},
      year={2020},
      eprint={2009.04806},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```