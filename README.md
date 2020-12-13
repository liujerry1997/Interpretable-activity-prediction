# Interpretable-activity-prediction

We proposed two approaches to perform antibiotic molecule discovery and classification: 1. Simple Graph Neural Network(SGC): Applying graph neural network on atom/bond level features. 2. Fingerprint Vector Model: Applying conventional neural network on molecular level features (molecular fingerprint). We prepared a dataset of 2335 molecules with labels, and extracted atom/bond/molecular features from the dataset. Baseline models are built for both approaches. Our multi-layer perceptron model applied to molecular fingerprints is able to achieve a 85.8\% testing accuracy. On the opposite, our SGC model only reaches an accuracy of 67\%. We discovered the reason was that the dataset is unbalanced, with 95\% of samples in one label of the two. Since the data is collected by biochemistry experiments, the amount of data is also limited. We then applied techniques such as model ensemble and hybrid model, to try to compensate the limitation in data size. After adjusting hyperparameters and comparing performance of the two approaches, we concluded fingerprint vector model is more suitable and accurate than SGC in our experiment. 

# Dataset
"Training_set.csv" contains 2,335 rows and 4 columns:  "Mean_Inhibition,""SMILES," "Name," and "Activity." Mean inhibition is a score of anti-bactericidal ability. SMILESare the string representations of the molecules’ graph structure. The CSV file for the testing dataset  contains 162 rows and 6 columns: "Broad_ID," "Name," "SMILES," Pred_Score," "ClinTox (low = less predicted toxicity)," "Mean_Inhibition," and "Activity." Our training dataset is from the paper "A Deep Learning Approach to Antibiotic Discovery" (Stokes etal., 2020). We also preprocessed the data first and upload the .npy here, which could be used by the script directly.



## Finger print model




## Simple Graph Simple Graph Neural Network
Two excutable jupyter notebooks for SGC model and SGC ensemble model are inside of SGC/ folder. In addition, modularized code for SGC ensemble model is also prepared. It is separated into three modules: dataset.py, training.py, main.py. 




# Result










# Result
