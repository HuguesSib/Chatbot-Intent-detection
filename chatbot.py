from transformers import AutoTokenizer, \
                        AutoModelForSequenceClassification,\
                        BertTokenizer

from src.dataset import BertDataset
from src.models import BERTClassifier

from src.evaluate import evaluate_csv, get_output

import time
from src.init import Options

def chatbot(tokenizer, model):
    print("Welcome to the chatbot !")
    while True:
        print("\n- Illuin BOT : Please enter your message :")
        try:
            prompt = input("- USER : ")
        except KeyboardInterrupt:
            print("\n- Illuin BOT : Bye !")
            break

        start_time = time.time()

        inputs = BertDataset([prompt], [''], tokenizer, max_length=128)
        string_label = get_output(inputs, model)

        end_time = time.time()
        response_time = end_time - start_time

        print('- Illuin BOT : The intent is :', string_label)

        if string_label == 'lost_luggage':
            print('- Illuin BOT : Vous avez perdu vos bagages ? Je vous redirige vers un agent ! (Notez que ce service vous sera facturé)')
        
        print(f'- Illuin BOT : Response time: {response_time:.2f} seconds')

if __name__ == "__main__":
    opt = Options().parse()

    if opt.model_name == 'bert-base-uncased' and opt.finetuned:
        tokenizer = BertTokenizer.from_pretrained(opt.model_name)
        model = BERTClassifier(opt.model_name, num_classes= opt.num_classes, freeze_bert = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(opt.model_name)
    
    if opt.eval:
        evaluate_csv(opt.csv_path, tokenizer, model)
    else:
        chatbot(tokenizer, model)
    