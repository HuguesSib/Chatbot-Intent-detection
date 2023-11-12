import pandas as pd
import torch
from src.dataset import BertDataset
from src.models import BERTClassifier
from transformers import BertTokenizer

def evaluate(path_to_csv, path_to_model, model_name):
    df = pd.read_csv(path_to_csv)

    model = BERTClassifier(model_name, num_classes = 9)
    model.load_state_dict(torch.load(path_to_model))

    prompts = df['text'].tolist()
    intents = df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = BertDataset(prompts, intents, tokenizer, max_length = 128)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['intent']

            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.argmax(1).cpu().tolist())
            labels.extend(label.cpu().tolist())
    
    return preds, labels

def get_accuracy(preds, labels):
    return sum([1 if pred == label else 0 for pred, label in zip(preds, labels)])/len(preds)


if __name__ == '__main__':
    preds, labels = evaluate('data/intent-detection-train.csv', 
                            'models/best_model.pt',
                            'bert-base-uncased')
    print(get_accuracy(preds, labels))