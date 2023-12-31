import pandas as pd
import torch
import json
import time
from src.dataset import BertDataset
from src.models import BERTClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

def postprocess(label: str):
    """
    Postprocesses the predicted label.
    """
    
    classes_of_interest = [
        'translate',
        'travel_alert',
        'flight_status',
        'lost_luggage', # /!\ to this class
        'travel_suggestion',
        'carry_on',
        'book_flight',
        'book_hotel',
        'oos', # might be removed because a binary classification problem
        ]
    
    if label not in classes_of_interest:
        label = 'oos'
    if label == 'oos':
        label = 'out_of_scope'

    return label

def get_output(inputs: BertDataset, model):
    """
    Returns the predicted intent of a given input text.
    """
    
    dataloader = torch.utils.data.DataLoader(inputs, batch_size=32, shuffle=True)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = model(input_ids, attention_mask)
            
            if isinstance(model, BERTClassifier):
                best_output = outputs.argmax(1).cpu().tolist()[0]
            else:
                best_output = outputs.logits.argmax(1).cpu().tolist()[0]
            break

    if isinstance(model, BERTClassifier):
        with open('data/labels_dict.json', 'r') as f:
            labels_dict = json.load(f)
        best_output = labels_dict[str(best_output)]
    else: 
        best_output = model.config.id2label[best_output]

    best_output = postprocess(best_output)
    return best_output

def evaluate_csv(csv, tokenizer, model):
    """
    Evaluates a model on a given csv dataset.
    """
    #Load a csv dataset contatining 'text' and 'label' columns
    df = pd.read_csv(csv)
    prompts = df['text'].tolist()
    intents = df['label'].tolist()
    
    predictions = []
    actual_labels = []
    inference_times = []

    for i, prompt in enumerate(prompts):
        to_dataset = BertDataset([prompt], [''], tokenizer, max_length = 128)
        start_time = time.time()
        output = get_output(to_dataset, model)
        end_time = time.time()
        inference_time = end_time - start_time
        label = intents[i]
        predictions.append(output)
        actual_labels.append(label)
        inference_times.append(inference_time)

    report = classification_report(actual_labels, predictions)
    avg_inference_time = sum(inference_times) / len(inference_times)
    logging.info("Below is the classification report :")
    print(report)
    logging.info(f"Average inference time: {avg_inference_time:.4f} seconds.")


if __name__ == '__main__':
    data = 'data/intent-detection-train.csv'
    model_name = 'lewtun/roberta-large-finetuned-clinc'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)