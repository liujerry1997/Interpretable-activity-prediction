# Interpretable-activity-prediction

We proposed two approaches to perform antibiotic molecule discovery and classification: 1. Simple Graph Neural Network(SGC): Applying graph neural network on atom/bond level features. 2. Fingerprint Vector Model: Applying conventional neural network on molecular level features (molecular fingerprint). We prepared a dataset of 2335 molecules with labels, and extracted atom/bond/molecular features from the dataset. Baseline models are built for both approaches. Our multi-layer perceptron model applied to molecular fingerprints is able to achieve a 85.8\% testing accuracy. On the opposite, our SGC model only reaches an accuracy of 67\%. We discovered the reason was that the dataset is unbalanced, with 95\% of samples in one label of the two. Since the data is collected by biochemistry experiments, the amount of data is also limited. We then applied techniques such as model ensemble and hybrid model, to try to compensate the limitation in data size. After adjusting hyperparameters and comparing performance of the two approaches, we concluded fingerprint vector model is more suitable and accurate than SGC in our experiment. 

## Finger print model




## Simple Graph Simple Graph Neural Network
