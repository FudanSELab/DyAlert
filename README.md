# DyAlert: Dynamic Graph Neural Networks-based Alert Link Prediction for Online Service Systems
This repository contains a Pytorch implementation of DyAlert.

DyAlert is a dynamic graph neural network-based approach, which takes alert propagation information into account for accurate alert linking. 
Specifically, we design a discrete-time dynamic graph (namely **A**lert-**M**etric **D**ynamic **G**raph) to describe the alert propagation process in DyAlert. Based on the dynamic graph, DyAlert uses heterogeneous k-GNNs to learn alert spatial information, and GRU to capture the temporal information of each alert within its active time. DyAlert predicts the links among alerts according to their spatio-temporal representations.


## Repository Organization
- **data_generator.py**: Generating example json data randomly, whose results do not correspond to the experimental results in our paper.
- **heterogeneous_graph_data_construction.py**: Constructing AMDG snapshots based on json files and generating pt files to be fed into model.
- **model/** contains:
  - **bert.py**: For alert embedding.
  - **customized_dataloader.py**: The dataloader implementation for DyAlert.
  - **focal_loss.py**: The implementation of focal loss.
  - **model.py**: The implementation of DyAlert model.
  - **sliding_dataset.py**: The implementation of sliding window, which is used for partitioning the dataset based on the occurrence time of alerts.
- **main_train.py**: Training the model of DyAlert with generated training pt files.
- **main_test.py**: Predict links among alerts in testing pt files based on the trained model.
