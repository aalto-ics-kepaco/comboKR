# comboKR

comboKR for predicting drug combination response surfaces

## System requirements

The code is developed with python 3.8. 

The main algorithm in comboKR.py is run with numpy 1.23.5, and scikit-learn 1.0.2.
The demo depends additionally on some other usual python packages, such as pandas and matplotlib. 

## Installation guide

There are two options for installing the comboKR package. Before installing the comboKR package make sure that latest versions of pip and build are installed:

>`pip3 install --upgrade pip`

>`pip3 install --upgrade build`

### Directly from the github

>`pip3 install git+https://github.com/aalto-ics-kepaco/comboKR.git#egg=comboKR`

### Downloading from github

>`mkdir comboKR`

>`cd comboKR`

>`git clone https://github.com/aalto-ics-kepaco/comboKR`

After downloading the comboKR package, it can be installed by the following command from the comboKR directory:

>`pip3 install .`

### Using comboKR

After installation comboKR can be imported as

>`from comboKR import ComboKR`


## Demo

A small-scale demo is provided in demo.py. It runs the experimental setup used in PIICM modification comparison experiments: see supplementary material. The expected runtime is as reported there: less than 10 seconds for one training-test cycle. 

## Instructions for use

The algorithm is implemented in the class ComboKR, implementing train and predict -methods. Example on how to use the algorithm can be found from the demo. 

