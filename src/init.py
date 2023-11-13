import argparse

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--finetuned", action="store_true",
                            help="Whether to load a finetuned model or not")
        
        parser.add_argument("--model_name", type=str, 
                        help="Pretrained model to use", 
                        default= 'lewtun/roberta-large-finetuned-clinc')
                        #transformersbook/bert-base-uncased-finetuned-clinc

        parser.add_argument("--eval", action="store_true", 
                            help="Evaluate the model on a dataset")
        
        parser.add_argument("--csv_path", type=str, 
                            help="Path to the csv file to evaluate the model on", 
                            default= 'data/intent-detection-train.csv')
        
        parser.add_argument("--model_path", type=str,
                            help="Path to the model to evaluate",
                            default= 'models/bert_model_classifier.pt')
        
        parser.add_argument("--num_classes", type=int,
                            help="Number of classes to classify",
                            default= 9)
        
        self.initialized = True
        return parser
    
    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(description="CLI Chatbot - Intent Detection")
            parser = self.initialize(parser)
        opt = parser.parse_args()
        
        return opt