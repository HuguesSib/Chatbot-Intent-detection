# Chatbot-Intent-detection
* [Getting started](#getting-started)
    * [Description](#description)
    * [Data](#data)
* [Instructions](#instructions)
    * [Requirements](#requirements)
    * [Architecture](#architecture)
    * [Run the project](#run-the-project)
* [Results](#results)
* [References](#references)
* 
# Getting Started

## Description

This project aims to create a chatbot that detects the intention of an input verbatim (prompt) for specific classes and in the presence of 'out of scope' prompts. The latter is output when the prompts don't belong to any given classes. 
The Chatbot uses models pre-trained on a common intent detection dataset: CLINC150 Introduced by Larson et al. in *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* [1]

## Data 

A CSV file was given with 75 examples (in French) of some prompts and their respective intention. Below is a distribution of this dataset. 

![alt text](/img/class_distib.png)

This set of prompts is too small to train a model on. A common dataset for intent classification tasks is CLINC150, especially since the classes in the CSV are also present in this dataset and it was thought to handle 'out-of-scopes' scenarios.

Some things are still to be considered: 
  - CLINC150 contains 150 in-scope intents, but we only consider 8. The others need to be classified as out-of-scope.
  - CLINC150 is in English, therefore a translation on-the-fly is chosen to be able to prompt in French and in any other language.
  - For evaluation the class _lost_luggage_
    
# Instructions 
## Requirements
## Architecture 
<pre>
Chatbot-Intent-detection/
┣ data/
┃ ┣ data_full.json
┃ ┣ data_oos_plus.json
┃ ┣ data_small.json
┃ ┣ intent-detection-train.csv
┣ notebooks/
┃ ┣ data_exploration.ipynb
┃ ┗ train_bert.ipynb
┣ src/
┃ ┣ dataset.py
┃ ┣ evaluate.py
┃ ┣ init.py
┃ ┗ models.py
┣ .gitignore
┣ chatbot.py
┣ README.md
┗ train.py
</pre>

## Run the project

# Results

# REFERENCES

[1] https://arxiv.org/pdf/1909.02027v1.pdf



