
import torch
from torch import nn
from transformers import DistilBertModel, AutoTokenizer

class NewsClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {"logits": logits}
    
    def prepare_input(self, texts):
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs
