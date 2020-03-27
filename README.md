# PerceptualAudio_pytorch

Pytorch implementation of "A Differentiable Perceptual Audio Metric Learned from Just Noticeable Differences", Pranay Manocha et al. - unofficial work in progress

Official repository in TensorFlow at: https://github.com/pranaymanocha/PerceptualAudio

current code =
* models
* training
* accuracy evaluation
* average perceptual distance evaluation
* loading of some pretrained models

data shoul be preprocessed as numpy dictionnaries in the format data_path+subset+'_data.npy'

subset in ['dataset_combined','dataset_eq','dataset_linear','dataset_reverb']

each entry is [first signal, second signal, human label]

target test loss is around 0.55 ~ 0.5

best result so far is test loss around 0.557 on the subset 'dataset_linear'

"experimental" features (as in the parser of train.py):
* dist_act = applies a non-linear activation to the distance output (e.g. some compression or expansion)
* classif_BN = selects which hidden layers of the classifier has Batch-Normalization
* classif_act = applies some compression to the classifier output (tends to reduce the overfitting)
* randgain = applies random gains to the audio pairs for training (to encourage invariance to audio level and apply the model on audio datasets with various gains)



