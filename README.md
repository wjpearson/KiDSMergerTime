# Determining the time before or after a galaxy merger event
## William J. Pearson
### B. Margalef Bentabol, V. Rodriguez-Gomez, L. Wang, L. E. Suelves

Code to accompany Pearson et al. A&A submitted (2024)

## Data

We use images identified as having merged in the last 500 Myr or will merge in the next 1000 Myr from IllustrisTNG 100 and 300 from snapshots 87 to 93 (inclusive). We further refine the merger time using simple gravity simulations, treating each merging galaxy as a single point mass. The images have a size of 128 x 128 pixels, an angular resolution of 0.2 arcsec/pixel, and have four channels: u, g, r, and i bands of KiDS. Each image also has a segmentation map for each band. Please see the paper for full details and note the images are not provided in this repo.

The training/validation/testing images should be saved in the form: `<object_name>.<time_to_merger in Myr>.fits`    
For example: `88_broadband_448870_xy.-432.fits`    
These should be placed in the `./data/train/`, `./data/valid/`, and `./data/test/` directories.
  

## Architectures

We use the CNN of [Pearson et al. A&A, 687, A45 (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...687A..45P/abstract). This CNN has  with six convolutional layers, three fully connected (dense) layers, and a single output neuron with sigmoid activation. The convolutional layers have 32, 64, 128, 256, 512, and 1024 filters with a size of 6, 5, 3, 3, 2, and 2 pixels, respectively, stride 1 and "same" padding. The dense layers have 2048, 512, and 128 neurons. The convolutional layers are followed by batch normalisation, dropout with a rate of 0.2, and 2 x 2 max-pooling. The dense layers are followed by batch normalisation and dropout with a rate of 0.1. The input is a four channel 128 x 128 pixel image using u, g, r, and i bands of our TNG images. The network is trained with MSE loss using the Adam optimiser.


## Running the scripts

To retrain the network, run the ecripts in order. Start with `cnn_wSegmap_step1_multiGPU.py` first, followed by `cnn_wSegmap_step2_multiGPU.py`, and so on.

## Acknowledge us

If you use these networks, please cite our paper:

```

```

## Acknowledgements

W.J.P. has been supported by the Polish National Science Center projects UMO-2020/37/B/ST9/00466 and UMO-2023/51/D/ST9/00147.    
The IllustrisTNG simulations were undertaken with compute time awarded by the Gauss Centre for Supercomputing (GCS) under GCS Large-Scale Projects GCS-ILLU and GCS-DWAR on the GCS share of the supercomputer Hazel Hen at the High Performance Computing Center Stuttgart (HLRS), as well as on the machines of the Max Planck Computing and Data Facility (MPCDF) in Garching, Germany.