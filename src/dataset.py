import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import string
from googletrans import Translator
from langdetect import detect


CLASSES_OF_INTEREST = [
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

class CLNIC150(Dataset):
    def __init__(self, path, set = 'train'):
        super().__init__()
        self.set = set
        self.path = path
        self.prompts, self.intents = self._read_clinc()

    def __len__(self):
        return len(self.prompt)
    
    def __getitem__(self, idx):
        return self.prompt[idx], self.intent[idx]
    
    def _read_clinc(self):

        #Load CLINC150 dataset from JSON file
        #Json from https://github.com/clinc/oos-eval/tree/master/data

        with open(self.path, 'r') as f:
            data = json.load(f)
        
        data = data[self.set]

        prompts = []
        intents = [] 

        for row in data:
            prompts.append(row[0])
            intents.append(row[1])

        return prompts, intents
    
    def _get_classes_of_interest(self, classes_of_interest: list = CLASSES_OF_INTEREST):
        #corpus.intent is a list 
        interest_index = []
        for i, intent in enumerate(self.intents):
            if intent in classes_of_interest:
                interest_index.append(i)

        self.intents = [self.intents[i] for i in interest_index]
        self.prompts = [self.prompts[i] for i in interest_index]
        return self.prompts, self.intents

class BertDataset(Dataset):
    def __init__(self, prompts, intents, tokenizer, max_length):
        super().__init__()
        #Preprocess the inputs prompts
        self.prompts = [self._preprocess(prompt) for prompt in prompts]
        
        #Convert intents to numeric labels
        self.labels_dict = {label: i for i, label in enumerate(set(intents))}
        self.intents = [self.labels_dict[intent] for intent in intents]

        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        intent = self.intents[idx]

        encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'prompt': prompt,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent': torch.tensor(intent, dtype=torch.long)
        }

    def _lower(self, text):
        return text.lower()
    
    def _remove_punctuation(self, text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)
    
    def _translate(self, text):
        lang = detect(text)
        if lang != 'en':
            translator = Translator()
            text = translator.translate(text, dest='en', str = 'auto').text
        return text

    def _preprocess(self, text):
        text = self._translate(text)
        text = self._lower(text)
        text = self._remove_punctuation(text)
        return text

if __name__ == '__main__':
    path = 'data/data_full.json'
    clinc_train = CLNIC150(path, train = True)
    clinc_test = CLNIC150(path, train = False)
    print(clinc_train[0])
    print(clinc_test[0])