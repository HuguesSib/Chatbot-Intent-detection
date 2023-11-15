from transformers import AutoTokenizer, \
                        AutoModelForSequenceClassification,\
                        BertTokenizer

from src.dataset import BertDataset
from src.models import BERTClassifier

from src.evaluate import evaluate_csv, get_output

import time
from src.init import Options
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

import time
from src.dataset import BertDataset
from src.evaluate import get_output
def chatbot(tokenizer, model):
    print('\n==================================================\n\tBienvenue sur le chatbot d\'Illuin!')
    while True:
        try:
            prompt = input("\n- Illuin BOT: \tVeuillez entrer votre demande :\n- VOUS:\t\t")
        except KeyboardInterrupt:
            print("\n- Illuin BOT: \tAu revoir!\n==================================================")
            break
        if prompt.lower() == 'exit':
            print("\n- Illuin BOT: \tAu revoir!\n==================================================")
            break
        start_time = time.time()
        inputs = BertDataset([prompt], [''], tokenizer, max_length=128)
        string_label = get_output(inputs, model)
        end_time = time.time()
        response_time = end_time - start_time
        print(f'- Illuin BOT: \tVotre demande concerne : {string_label}')
        if string_label == 'lost_luggage':
            print('- Illuin BOT: \tVous avez perdu vos bagages ? Je vous redirige vers un agent ! (Notez que ce service vous sera facturé)')
        print(f'- Illuin BOT: \tTemps de reponse {response_time:.2f} secondes')

if __name__ == "__main__":
    opt = Options().parse()

    if opt.finetuned:
        #For trained model BERTClassifier locally (need a {model}.pt file)
        #model_name is the pretrained encoding part of the model --> only trained with 'bert-base-uncased'
        #TODO : Train another model with finetuned encoder on clinc dataset

        tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
        model = BERTClassifier(opt.model_name, num_classes= opt.num_classes, freeze_bert = True)
    else:
        #For pretrained model doing Sequence Classification from HuggingFace (no need to train)
        #model_name is the pretrained model
        #tried : 'bert-base-uncased-finetuned-clinc' (from transformersbook)
        #        'distilbert-base-uncased-distilled-clinc' (from transformersbook)
        #        'lewtun/roberta-large-finetuned-clinc' (from lewtun)
        
        tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(opt.model_name)
    
    if opt.eval:
        logging.info(f"Evaluating {opt.model_name} on test data : {opt.csv_path}...")
        evaluate_csv(opt.csv_path, tokenizer, model)
    else:
        logging.info("Starting chatbot...")
        chatbot(tokenizer, model)
    