# FedDef
Code for our paper-FedDef: Robust Federated Learning-based Network Intrusion Detection Systems Against Gradient Leakage
See example.txt for example running, including training FL-based NIDS, leveraging reconstruction attacks to obtain privacy score and GAN-based adversarial attack
data denotes our dataset, gan_dataset denotes the reconstructed private data by reconstruction attack.
Note that the (x, y) pairs under gan_dataset directory represent the results with single sample reconstruction, while pairs in save_bs_5 represent the samples with batch=5 reconstruction attack.
kitsune_model includes the trained Kitsune model for evaluating evasion rate against Kitsune.
save denotes the saved privacy score during reconstruction attack, while save_model includes our trained FL-based NIDS with different defense strategies.
model includes some codes for federated learning.
Autoencoder.py, corClust.py and KitNET.py denote partial Kitsune codes to evaluate Kitsune model's classification for adversarial traffic. 