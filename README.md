# Chatbot-Intent-detection
* [Getting started](#getting-started)
    * [Description](#description)
    * [Data](#data)
* [Instructions](#instructions)
    * [Architecture](#architecture)
    * [Run the project](#run-the-project)
       * [Clone the Project](#clone-the-project)
       * [Install Requirements](#install-requirements)
       * [Download dataset](#download-dataset-optional)
       * [Run Chatbot](#run-chatbot)
       * [Evaluation on a CSV](#evaluation-on-a-csv-file)
       * [Training](#train-a-bertclassifier-to-be-improved)
* [Results](#results)
* [References](#references)
  
# Getting Started

## Description

This project aims to create a chatbot that detects the intention of an input verbatim (prompt) for specific classes and in the presence of 'out of scope' prompts. The latter is output when the prompts don't belong to any given classes. 
The Chatbot uses models pre-trained on a common intent detection dataset: CLINC150 Introduced by Larson et al. in this paper[[1](#references)]

Below is a snapshot of a conversation with the chatbot on few questions and in different languages.

![alt text](/img/chatbot.png)

*Note:* 
|Original| Translation|
|-|-|
| kan jag boka ett flyg till Paris imorgon? | can i book a flight to paris tomorrow    |
| 高中毕业有什么好处                         | What are the benefits of graduating from high school?    |
| 我把我的行李弄丢了                          |  I lost my baggage       |

## Data 

A CSV file was given with 75 examples (in French) of some prompts and their respective intention. Below is a distribution of this dataset. 

![alt text](/img/class_distib.png)

This set of prompts is too small to train a model on. A common dataset for intent classification tasks is CLINC150, especially since the classes in the CSV are also present in this dataset and it was thought to handle 'out-of-scopes' scenarios.

Some things are still to be considered: 
  - CLINC150 contains 150 in-scope intents, but we only consider 8. The others need to be classified as out-of-scope. 
  - CLINC150 is in English, therefore a translation on the fly is chosen to be able to prompt in French and any other language.The translation is done with Google Translate free API accessible through the library [*googletrans*](https://pypi.org/project/googletrans/)
  - For evaluation, the class _lost_luggage_ needs to have a low number of False Positive (High Precision). 
    
# Instructions 

## Architecture 
```bash
Chatbot-Intent-detection/
┣ data/                         --different .json dataset of CLINC
┃ ┣ data_full.json            
┃ ┣ data_oos_plus.json
┃ ┣ data_small.json
┃ ┣ intent-detection-train.csv  --test samples to evaluate the model
┣ notebooks/
┃ ┣ data_exploration.ipynb
┃ ┗ train_bert.ipynb
┣ src/
┃ ┣ dataset.py                  --create datasets and preprocess
┃ ┣ evaluate.py                 --evaluation of a model given a csv  
┃ ┣ init.py                     --arguments init for CLI
┃ ┗ models.py                   --bert classifier class
┣ .gitignore
┣ chatbot.py                    --script to run the chatbot or evaluation
┣ README.md
┗ train.py                      --script to train a finetuned model
```
## Run the project
### Clone the project 
```bash
git clone https://github.com/HuguesSib/Chatbot-Intent-detection.git
cd Chatbot-Intent-detection
```
### Install requirements 
```bash
conda env create -f environment.yml
conda activate chatbot
```

### Download dataset [OPTIONAL]
If needed install json dataset and put it in the ./data folder. 
Some dataset from the CLINC150 are already present and were download from https://github.com/clinc/oos-eval/tree/master/data

### Run Chatbot 
To see the optional parameter to run the CLI please run
```bash
python chatbot.py -h
```
If you want to run the chatbot with a given model 

```bash
python chatbot.py --model_name {name}
```
### Evaluation on a CSV file
To evaluate how good the chatbot perform with a given model use 
```bash
python chatbot.py --eval --model_name {name}
```
This will print the classification report from the model and the average inference time. 

### Train a BertClassifier [to be improved]
If you have a GPU you can train your finetuned bert classifier, consisting of a pre-trained bert encoding part and a linear layer to classify the outputs.

You can see the parameters with 
```bash
python train.py -h
```
and train it. 
```bash
python train.py --epochs 50 --batch_size 32 --lr 2e-4
```

# Results
## Comparison of different pre-trained BERT model on the CLINC dataset
Here is an over view of the results from the different pretrained model.

| Model Name                                                | Avg Precision | Avg Recall | Avg F1-Score | Precision on Lost Luggage | Inference Time (s) |
|-----------------------------------------|-----------------|------------|--------------|---------------------------|---------------------|
| bert-base-uncased-finetuned-clinc [[2](#references)]      | 0.96          | 0.96       | 0.96         | 0.88                      | 0.3706              |
| roberta-large-finetuned-clinc [[3](#references)]          | 0.95          | 0.95       | 0.95         | 1.00                      | 1.2057              |
| distilbert-base-uncased-distilled-clinc [[4](#references)] | 0.96          | 0.96       | 0.96         | 0.88                      | 0.1693              |

And below are the details of the classification report for each model. 

### transformersbook/bert-base-uncased-finetuned-clinc

| Intent             | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight        | 1.00      | 1.00   | 1.00     | 6       |
| book_hotel         | 1.00      | 1.00   | 1.00     | 7       |
| carry_on           | 1.00      | 0.88   | 0.93     | 8       |
| flight_status      | 1.00      | 1.00   | 1.00     | 6       |
| lost_luggage       | 0.88      | 1.00   | 0.93     | 7       |
| out_of_scope       | 0.91      | 1.00   | 0.95     | 21      |
| translate          | 1.00      | 0.86   | 0.92     | 7       |
| travel_alert       | 1.00      | 1.00   | 1.00     | 5       |
| travel_suggestion  | 1.00      | 0.88   | 0.93     | 8       |
| **Micro Avg**      | **0.96**  | **0.96**| **0.96** | **75**  |
| **Macro Avg**      | **0.98**  | **0.96**| **0.96** | **75**  |
| **Weighted Avg**   | **0.96**  | **0.96**| **0.96** | **75**  |

Average inference time: 0.3706 seconds.

### lewtun/roberta-large-finetuned-clinc

| Intent             | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight        | 1.00      | 1.00   | 1.00     | 6       |
| book_hotel         | 1.00      | 1.00   | 1.00     | 7       |
| carry_on           | 1.00      | 0.88   | 0.93     | 8       |
| flight_status      | 0.86      | 1.00   | 0.92     | 6       |
| lost_luggage       | 1.00      | 1.00   | 1.00     | 7       |
| out_of_scope       | 0.91      | 0.95   | 0.93     | 21      |
| translate          | 1.00      | 1.00   | 1.00     | 7       |
| travel_alert       | 0.83      | 1.00   | 0.91     | 5       |
| travel_suggestion  | 1.00      | 0.75   | 0.86     | 8       |
| **Micro Avg**      | **0.95**  | **0.95**| **0.95** | **75**  |
| **Macro Avg**      | **0.96**  | **0.95**| **0.95** | **75**  |
| **Weighted Avg**   | **0.95**  | **0.95**| **0.95** | **75**  |

Average inference time: 1.2057 seconds.

### transformersbook/distilbert-base-uncased-distilled-clinc

| Intent             | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| book_flight        | 1.00      | 1.00   | 1.00     | 6       |
| book_hotel         | 1.00      | 1.00   | 1.00     | 7       |
| carry_on           | 1.00      | 0.88   | 0.93     | 8       |
| flight_status      | 1.00      | 1.00   | 1.00     | 6       |
| lost_luggage       | 0.88      | 1.00   | 0.93     | 7       |
| out_of_scope       | 0.91      | 1.00   | 0.95     | 21      |
| translate          | 1.00      | 0.86   | 0.92     | 7       |
| travel_alert       | 1.00      | 1.00   | 1.00     | 5       |
| travel_suggestion  | 1.00      | 0.88   | 0.93     | 8       |
| **Micro Avg**      | **0.96**  | **0.96**| **0.96** | **75**  |
| **Macro Avg**      | **0.98**  | **0.96**| **0.96** | **75**  |
| **Weighted Avg**   | **0.96**  | **0.96**| **0.96** | **75**  |

Average inference time: 0.1693 seconds.

## Finetuning and training of a bert classifier model
Finally I tried to train a bert classifier with a linear layer on the top of it to correctly classify out classes of interests. 

Here are the training plots for the following command (trained on google Colab GPU); details can be found in *./notebooks/train_bert.ipynb*

```bash
python train.py --model_name bert-base-uncased --json_path data/data_full.json --epochs 500 --lr 2e-4 --batch_size 32 --patience 50
```

![alt text](/img/loss_accuracy.jpg)

And here are the classification report from the validation dataset after the training

![alt text](/img/report_val.jpg)



# REFERENCES

[1] [Larson, et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.](https://arxiv.org/pdf/1909.02027v1.pdf)

[2] [bert-base-uncased finetuned model](https://huggingface.co/transformersbook/bert-base-uncased-finetuned-clinc)

[3] [Liu, et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. Submitted on 26 Jul 2019.](https://arxiv.org/pdf/1907.11692.pdf)

[4] [Sanh, V., et al. (2020). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.](https://arxiv.org/pdf/1910.01108.pdf)





