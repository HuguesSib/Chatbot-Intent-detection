import argparse

class Options():
    """
    Class that defines the options for the project.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--finetuned", action="store_true",
                            help="Whether to load a finetuned model or not")
        
        parser.add_argument("--model_name", type=str, 
                        help="Pretrained model to use", 
                        default= 'transformersbook/distilbert-base-uncased-distilled-clinc')
                        #transformersbook/bert-base-uncased-finetuned-clinc
                        #transformersbook/distilbert-base-uncased-distilled-clinc
                        #lewtun/roberta-large-finetuned-clinc
    
        parser.add_argument("--eval", action="store_true", 
                            help="Evaluate the model on a csv dataset")
        
        parser.add_argument("--csv_path", type=str, 
                            help="Path to the csv file to evaluate the model on", 
                            default= 'data/intent-detection-train.csv')
        
        parser.add_argument("--model_path", type=str,
                            help="Path to the model to evaluate",
                            default= 'models/bert_model_classifier.pt')
        
        parser.add_argument("--num_classes", type=int,
                            help="Number of classes to classify",
                            default= 9)
        
        #Initialize argument for training
        parser.add_argument("--json_path", type=str,
                            help="Path to the json file to train the model on",
                            default= 'data/data_full.json')
        
        parser.add_argument("--batch_size", type=int,
                            help="Batch size",
                            default= 32)
        
        parser.add_argument("--epochs", type=int,
                            help="Number of epochs",
                            default= 1)
        
        parser.add_argument("--lr", type=float,
                            help="Learning rate",
                            default= 2e-4)
        
        parser.add_argument("--patience", type=int,
                            help="Patience for early stopping",
                            default= 30)
        

        self.initialized = True
        return parser
    
    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(description="CLI Chatbot - Intent Detection")
            parser = self.initialize(parser)
        opt = parser.parse_args()
        
        return opt