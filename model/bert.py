from transformers import BertTokenizer, BertModel
import torch
import warnings

warnings.filterwarnings('ignore')


class Bert():
    def __init__(self, tokenizer_pretrained='hfl/chinese-bert-wwm-ext', model_pretrained='hfl/chinese-bert-wwm-ext'):
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.bert_model = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')

    def get_batch_sentence_embedding(self, batch_sentence, max_length=128):
        inputs = self.tokenizer.batch_encode_plus(
            batch_sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output


