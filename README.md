# PerceptualAudio_pytorch

Pytorch implementation of "A Differentiable Perceptual Audio Metric Learned from Just Noticeable Differences", Pranay Manocha et al. - unofficial work in progress

Official repository in TensorFlow at: https://github.com/pranaymanocha/PerceptualAudio

current code =
* models
* training
* accuracy evaluation
* average perceptual distance evaluation

data shoul be preprocessed as numpy dictionnaries in the format data_path+subset+'_data.npy'

subset in ['dataset_combined','dataset_eq','dataset_linear','dataset_reverb']

each entry is [first signal, second signal, human label]

target test loss is around 0.55 ~ 0.5

best result so far is around 0.57
