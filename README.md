# AF Classification with Regression

## Overview

This repository contains the code and resources for a research project focused on classifying atrial fibrillation (AF) episodes using regression-based prediction. The project aims to improve AF classification by predicting the time until termination and then classifying the episodes based on this prediction.

## Installation

To use this project, follow these steps:

1. Clone the repository to your local machine:
   git clone https://github.com/HodayaRabinovich/AF-classification.git
2. Install requirement: biosppy.signals.ecg
3. download the database from "https://physionet.org/content/challenge-2004/1.0.0/"

## Usage
There are 2 modes:
   ### test: 
      python main.py -m test -n model
   ### train:
      python main.py -m train -n "new-model"
      python main.py -m test -n "new-model"

For results and conclusions, refer to the documentation in the docs directory.

## References

1. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
2. https://github.com/antonior92/ecg-age-prediction

3. https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-eeg
