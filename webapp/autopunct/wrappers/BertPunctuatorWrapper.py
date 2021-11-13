
import numpy as np
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from autopunct.base.BaseWrapper import BaseWrapper
from autopunct.models.BertPunctuator import BertPunctuator
#from trainers.BertPunctuatorTrainer import BertPunctuatorTrainer


class BertPunctuatorWrapper(BaseWrapper):
    def __init__(self, config, checkpoint):
        super().__init__(config)

        self._classifier = BertPunctuator(config)
        self._classifier.load_state_dict(checkpoint['model_state_dict'])
        self._classifier.eval()

        self._tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        #self._trainer = BertPunctuatorTrainer(self._classifier, self._preprocessor, self._config)

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
        #mask &=  (targets != -1).type(torch.uint8)
        mask = list(map(bool, mask))

        text[mask] = mask_token_number
        return text, mask

    def predict(self,message):
        """
        To make a prediction on one sample of the text
        satire or fake news
        :return: a result of prediction in HTML page
        """

        #message = "Hey WorkLifers its Adam Grant I hope youre enjoying season four today I want to share a special bonus conversation with Glennon Doyle"
        data = self._tokenizer.encode(message)
        data_processed = []

        data_processed.append(data[0])
        for token in data[1:-1]: 
            data_processed.append(token)
            data_processed.append(6)
        data_processed.append(data[-1])

        vect, mask = self.full_mask_out(np.array(data_processed),self._tokenizer)
        vect = np.expand_dims(vect,axis=0)
        prediction,_ = self._classifier(torch.Tensor(vect).long())

        prediction = torch.squeeze(prediction)
        prediction = prediction[mask]

        prediction = np.exp(prediction.detach().cpu().numpy())
        prediction = prediction.reshape(-1,4)
        prediction = prediction.argmax(-1)

        punctArray = [' ','.','?',',']
        decoded_inputs  = self._tokenizer.convert_ids_to_tokens(vect[0])
        
        punctCount = 0
        for i in range(len(decoded_inputs)):
            if decoded_inputs[i] == '<mask>':
                decoded_inputs[i] = punctArray[prediction[punctCount]]
                punctCount+=1
        
        decode = []
        for i in decoded_inputs[1:-1]:
            if i != ' ':
                decode.append(i)
        
        text = self._tokenizer.decode(self._tokenizer.convert_tokens_to_ids(decode))
        return text
