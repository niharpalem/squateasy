# Squat Assistant with BiLSTM and ConvLSTM Models

"This project aims to develop and evaluate two deep learning models, BiLSTM and ConvLSTM, for squat assistance. The models are designed to analyze and provide feedback on squat form based on input data."

## Data
You can download the required data from the following link: [https://drive.google.com/drive/folders/1HFrkfbCdk21wYa3ANXRoOnLdv0jum22v?usp=sharing]

-- Make sure to extract the downloaded data (DATA, Divided) both into the project directory and make sure to organise the files in the below format.

## Directory Structure
project/
├── DATA
├── divided/
    ├── train.json
│ 
├── data_loader.py
├── model1.py
├── model2.py
├── main.py
├── requirements.txt
├── saved_models/
└── results/

data/: Contains the raw data files
divided/: Contains the train.json..
data_loader.py: Contains functions to load and preprocess data.
model1.py: Defines the BiLSTM model.
model2.py: Defines the ConvLSTM model.
main.py: Script to train and evaluate models.
requirements.txt: Lists the required Python packages.
saved_models/: Directory to store trained models.
results/: Directory to store training history plots.

## Installation
Make sure you have Python and pip installed on your system.
Navigate to the project directory.

-- Install the required packages by running: pip install -r requirements.txt

## Usage
-- EDA
    In order to see the Exploratory data analysis of the data run: main eda.py 
    (Note: all the plots will be saved in the new directory as eda
    where you can see the class distribution before and after splitting.
    
-- Training and Evaluating Models

    To train and evaluate the BiLSTM model, run the following command: python main.py --train bilstm --train_file Divided/train.json
    
    To train and evaluate the ConvLSTM model, run the following command: python main.py --train  --train_file Divided/train.json
    
    Explanation:
        --train: Specifies the model type to train. Use bilstm for the BiLSTM model or convlstm for the ConvLSTM model.
        --train_file: Specifies the path to the training data file (train.json).
    During training, the model will be saved in the saved_models/ directory, and the training history plot will be saved in the plots/ directory.
    Once the traning is done it works on the test set displays the results.

-- As there is another code file which is test_on_random_video in order to check this how the medaipipe works, run the code in the google colab with the saved model (as there is comptability issues with cv2 and python versions for mediapipe and we tried to resolve it but with colab it is done smoothly)

# Contributions:


| Teammate     | Contribution                     |
|--------------|----------------------------------|
| Nihar Reddy  | Project Planning                 |
|              | Literature Survey                |
|              | Data Collection                  |
|              | Modelling                        |
|              | Documentation                   |
|--------------|----------------------------------|
| Charishma    | Literature Survey                |
|              | Data Collection                  |
|              | Modelling                        |
|              | Documentation                   |
|--------------|----------------------------------|
| Mahesh       | Literature Survey                |
|              | Data Pre-Processing              |
|              | Modelling                        |
|              | Documentation                   |
|--------------|----------------------------------|
| Sai Krishna  | Literature Survey                |
|              | Data Preprocessing               |
|              | Modelling                        |
|              | Documentation                   |




        






