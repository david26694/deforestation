# deforestation

Code for the deforestation datathon

## Setup

Create virtual environment:
```bash
conda create -y --name deforestation python==3.9
conda activate deforestation
pip install -r requirements.txt
```

## Main file

The main file is `main.ipynb`. It contains the code for the model model training and ensemble. It can be run after installing the requirements and using the correct virtual environment, but it is slow since we are training a Neural network. For faster, training we recommend uploading the data and notebook to kaggle and running it from there, as we did during the hackathon.

It expects the data from the competition to be in the data folder, as provided by the organizers (a folder called train_test_data and two csv files, train.csv and test.csv).

## Huggingface app

The link to the huggingface app can be found in here: https://huggingface.co/spaces/davidmasip/Deforestation. 

You can see that the last commit is from Saturday 19th of November.

You can find some of the code of the app in `app/`. Not all code is included, as big files (the model.pkl) are not uploaded to github.


## Other files

We have some files in src of things that were tried in the hackathon, they contributed to the final solution but we prefered to have it all in one file.

We also have some files in pre-datathon for preparation for the datathon, we were mainly playing with image data of other datasets.