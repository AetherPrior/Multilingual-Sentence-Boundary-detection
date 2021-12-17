
import numpy as np
import torch
import jieba
from collections import OrderedDict
from transformers.models.auto.tokenization_auto import AutoTokenizer
from autopunct.base.BaseWrapper import BaseWrapper


class BertPunctuatorWrapper(BaseWrapper):
    def __init__(self, config, checkpoint):
        super().__init__(config)

        self._classifier = torch.jit.load(checkpoint)
        self._classifier.eval()

        self._tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    def full_mask_out(self,text,tokenizer):
        """
        mask every single punctuation mark
        """
        period = tokenizer.encode(".",add_special_tokens=False)[1]
        chinese_period = tokenizer.encode("｡",add_special_tokens=False)[1]
        comma = tokenizer.encode(",",add_special_tokens=False)[1]
        question = tokenizer.encode("?",add_special_tokens=False)[0]
        mask_token_number = tokenizer.encode('<mask>',add_special_tokens=False)[0]
        space_placeholder = 6

        values_mask = [period,comma,chinese_period,space_placeholder,question]
        mask = sum(text == i for i in values_mask)
        mask = list(map(bool, mask))

        text[mask] = mask_token_number
        return text, mask

    def predict(self,message, ptype='all'):
        """
        To make a prediction on one sample of the text
        :return: a result of prediction in HTML page
        """
        position = prev_end = 0
        lang_delim = OrderedDict()
        while position < len(message):
            if '\u4e00' <= message[position] <= '\u9fff':
                start_zh = position
                while(position < len(message) and ('\u4e00' <= message[position] <= '\u9fff' or message[position].isspace())): 
                    position+=1
                end_zh = position-1
                if start_zh!= 0:
                    lang_delim[(prev_end,start_zh-1)] = 'en'
                lang_delim[(start_zh,end_zh)] = 'zh'
                prev_end = end_zh+1
            position+=1
        if prev_end != len(message):
            lang_delim[(prev_end, len(message)-1)] = 'en'

        tokens = []
        for key,val in lang_delim.items():
            if val == 'en':
                tokens.extend(message[key[0]:key[1]+1].split(' '))
            else:
                tokens.extend(jieba.cut(message[key[0]:key[1]+1],HMM=True))

        encoded_texts = []
        
        tokens = [token for token in tokens  if len(self._tokenizer.encode(token)[1:-1]) > 0]
        
        for token in tokens:
            li = self._tokenizer.encode(token)[1:-1]
            if li and li[0] == 6:
                li = li[1:]
            encoded_texts.extend(li)
            encoded_texts.append(6)

        vect, mask = self.full_mask_out(np.array(encoded_texts),self._tokenizer)
        vect = np.expand_dims(vect,axis=0)
        prediction,_ = self._classifier(torch.Tensor(vect).long())

        prediction = torch.squeeze(prediction)
        prediction = prediction[mask]

        prediction = np.exp(prediction.detach().cpu().numpy())
        prediction = prediction.reshape(-1,4)
        prediction = prediction.argmax(-1)

        punctArray = [' ','.','?',','] if ptype == 'all' else [' ','.',' ',' ']
        decoded_inputs  = self._tokenizer.convert_ids_to_tokens(vect[0])
        
        punctCount = 0
        for i in range(len(decoded_inputs)):
            if decoded_inputs[i] == '<mask>':
                decoded_inputs[i] = punctArray[prediction[punctCount]]
                punctCount+=1
        
        decode = []
        for i in decoded_inputs:
            if i != ' ':
                decode.append(i)
        
        text = self._tokenizer.decode(self._tokenizer.convert_tokens_to_ids(decode))
        return text
