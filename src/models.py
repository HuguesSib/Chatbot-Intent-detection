from torch import nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, freeze_bert = True):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        #Add a fully-connected layer to the bert model for classification
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(output.pooler_output)
        logits = self.fc(output)
        return logits
