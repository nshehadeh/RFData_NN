# RFData_NN
## Project Proposal
This is an undergraduate research project done in Dr. Brett Byram's BEAM Lab in the Vanderbilt Institute for Surgery and Engineering. I proposed to build a deep neural network (DNN) to improve image quality by increasing the SNR of ultrasound data. Coded excitation is a recently developed technique used in ultrasound to improve SNR at the time of data collection. My neural network would effectively aim to apply the same effect as using coded excitation on past clinical data that was collected without coded excitation.
## Project Description
The current iteration of my DNN is a proof of concept network. It is feed forward and uses a ReLU activation function, MSE loss function, and Adam optimizer for SGD. Hyper parameters are read in through a YAML config file. The inputs are simulated phantoms with each phantom being a sample. The output is an image with the artifical filter applied. The architecture is designed to be customizable and allow for easy testing. 
## File Descriptions
### Main
- Takes in all parameters
- Sets up data loaders, concatenates data sets
- Creates and trains model
### Dataloader
- Iterates over all phantoms of one dataset
- Stores data in 3D tensor [phantom, beam, depth]
- Has functionality to read in data by any dimensional slice
- Main.py is set up to handle multiple datasets and concatenate the data
### Trainer
- Records training loss, training evaluation loss, and validation loss to logger
- Has ‘patience’ and max_epochs
### Model
- Connects input, hidden, and output
- Has options for dropout and batch normalization 
### Logger
- Writes losses to a file obtained by trainer
### Test_model
- Loads testing data set and model.dat
- Runs testing input through a model
- Outputs .mat file for further analysis (image quality metrics)

## Current State of Project, Initial Results
I ran five different models toward the end of the semester and acquired some initial results. They were a proof of concept for my network, and I recorded loss curves and image quality metrics for modeled images to compare to the testing data input. The results show it is possible to increase SNR with a DNN, but it will take more tuning and experimentation to have a more consistent model to use on old clinical data.
## Future Steps
In the future, I am going to work on hyperparameter tuning and trying other architecutres such as convolutional neural networks. The main hyper parameters I will target are layer width, number of hidden layers, and different loss functions and optimizers.
## Collaborators
Emelina Vienneau

Lab: Dr. Brett Byram

Advisor: Dr. Matthew Berger
## License
