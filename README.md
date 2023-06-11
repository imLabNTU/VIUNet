# VIUNet
This is the official implamentation of VIUNet from
*VIUNet: Deep Visual–Inertial–UWB Fusion for Indoor UAV Localization (IEEE ACCESS'23)*

## Quickstart
### Enviornment
The enviornment can be installed via conda
```sh
conda env create -f environment.yml
conda activate VIUNet
```
### Dataset formating

The dataset should be formated as in kitti format, you can do so with following command
```sh
python preprocess.py --dir ../euroc --output_dir ../test  --type Euroc
```

Then, run the UWB simulation to generate the UWB data
```sh
python simulate.py --dir ../test --type Euroc
```

### Training

First Download the pretrained model from [here](https://www.dropbox.com/s/7qgncgmwqlfc51t/euroc_test.tar.xz?dl=0) and unzip it into the `./pretrained` folder.
```sh
cd pretrained
wget 'https://www.dropbox.com/s/7qgncgmwqlfc51t/euroc_test.tar.xz?dl=0'
tar -xvf euroc_test.tar.xz
cd ..
```


Then run the following command to train the model

```sh
python train.py <path/to/dataset>
```
In default, the result will be saved at `./results/checkpoints` folder, tesnorboard log will be saved at `./runs` folder.


### Testing

# Realworld Dataset
We present a real world dataset, available [here](https://drive.google.com/drive/folders/18YPoqVJtSfsa2m2O1KJd5tHbk7FQlAww?usp=sharing)
