import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.dataset import BertDataset
from googletrans import Translator
from langdetect import detect

def _translate(text):
    lang = detect(text)
    if lang != 'en':
        translator = Translator()
        text = translator.translate(text, dest='en', str = 'auto').text
    return text

import time

def chatbot(tokenizer, model):
    print("Welcome to the chatbot !")
    while True:
        print("\n- Illuin BOT : Please enter your message :")
        try:
            prompt = input("- USER : ")
        except KeyboardInterrupt:
            print("\n- Illuin BOT : Bye !")
            break

        prompt = _translate(prompt)

        start_time = time.time()
        
        inputs = BertDataset([prompt], [''], tokenizer, max_length=128)
        dataloader = torch.utils.data.DataLoader(inputs, batch_size=32, shuffle=True)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                outputs = model(input_ids, attention_mask)

                best_output = outputs.logits.argmax(1).cpu().tolist()[0]
                
        end_time = time.time()
        response_time = end_time - start_time

        string_label = model.config.id2label[best_output]

        print('- Illuin BOT : The intent is :', string_label)
        print(f'- Illuin BOT : Response time: {response_time:.2f} seconds')

if __name__ == "__main__":
        # Define the CLI arguments
    parser = argparse.ArgumentParser(description="CLI Chatbot")
    parser.add_argument("--model_name", type=str, help="Pretrained model to use", 
                        default= 'lewtun/roberta-large-finetuned-clinc')

    # Parse the arguments
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    chatbot(tokenizer, model)
    