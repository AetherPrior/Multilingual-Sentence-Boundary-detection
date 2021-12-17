from time import time
import logging
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from neural_punctuator.base.BaseTrainer import BaseTrainer
from neural_punctuator.data.dataloader import BertDataset, collate, get_data_loaders, get_datasets_metrics
from neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW  # TODO
from torch_optimizer import RAdam
from torch import nn
import wandb
from torch.utils.tensorboard import SummaryWriter

from neural_punctuator.utils.data import get_target_weights
from neural_punctuator.utils.io import save, load
from neural_punctuator.utils.loss import WeightedBinaryCrossEntropy
from neural_punctuator.utils.metrics import get_total_grad_norm, get_eval_metrics
from neural_punctuator.utils.tensorboard import print_metrics
from neural_punctuator.utils.scheduler import LinearScheduler
import numpy as np

torch.manual_seed(69)
np.random.seed(69)
torch.backends.cudnn.deterministic = True


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


class BertPunctuatorTrainer(BaseTrainer):
    def __init__(self, model, preprocessor, config):
        super().__init__(model, preprocessor, config)

        if self._config.trainer.use_gpu:
            self.device = torch.device(self._config.trainer.use_gpu)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        model_family = self._config.model.load_model_repo.split('-')[0]
        
        if model_family == 'bert':
            self.mask_token = '[MASK]'
        elif model_family == 'xlm':
            self.mask_token = '<mask>'
        else:
            log.error("Please use a proper model")
            exit(1)
            
        self.train_dataset, self.valid_dataset, self.space_count, self.p_count, self.q_count, self.comma_count = get_datasets_metrics(config)
        self.punct_count = self.space_count + self.p_count + self.q_count + self.comma_count
 
        self.train_loader, self.valid_loader = get_data_loaders(self.train_dataset, self.valid_dataset, config)
        self.model = model.to(self.device)
        self.model.train()

        if self._config.trainer.loss == 'NLLLoss':
            target_weights = torch.Tensor(get_target_weights(self.train_dataset.targets,
                                                             self._config.model.num_classes)).clamp_max(1).to(self.device)
            self.criterion = nn.NLLLoss(weight=target_weights, reduction='none')
        else:
            log.error('Please provide a proper loss function')
            exit(1)

        optimizer_args = [
                {'params': self.model.base.parameters(), 'lr': self._config.trainer.base_learning_rate},
                {'params': self.model.classifier.parameters(), 'lr': self._config.trainer.classifier_learning_rate}
            ]
        optimizer_args_radam =[
                {'params': self.model.base.parameters(), 'lr': self._config.trainer.base_learning_rate, 'weight_decay': self._config.trainer.weight_decay},
                {'params': self.model.classifier.parameters(), 'lr': self._config.trainer.classifier_learning_rate, 'weight_decay': self._config.trainer.weight_decay}
            ]
        if self._config.trainer.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(optimizer_args)

        elif self._config.trainer.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(optimizer_args)
        elif self._config.trainer.optimizer == 'radam':
            self.optimizer = RAdam(optimizer_args_radam)
        else:
            log.error('Please provide a proper optimizer')
            exit(1)

        if self._config.trainer.load_model:
            self.epochs = load(self.model, self.optimizer, self._config)
        else:
            self.epochs = 0

        self.sched = LinearScheduler(self.optimizer, self._config.trainer.warmup_steps)

        self.all_valid_target = np.concatenate([targets.numpy() for _, targets in self.valid_loader])
        self.all_valid_target = self.all_valid_target[self.all_valid_target != -1]

        if self._config.debug.summary_writer:
            self.summary_writer = SummaryWriter(comment=self._config.experiment.name)
        else:
            self.summary_writer = None
    
    def mask_out(self,text,targets):
        """
        mask some punctuations 
        """
        period = self._preprocessor._tokenizer.encode(".")[1:-1]
        chinese_period = self._preprocessor._tokenizer.encode("｡")[1:-1]
        comma = self._preprocessor._tokenizer.encode(",")[1:-1]
        question = self._preprocessor._tokenizer.encode("?")[1:-1]
        mask_token_number = self._preprocessor._tokenizer.encode(self.mask_token)[1:-1]
        
        space_placeholder = 6
        
        mask =   ((text == space_placeholder) & (np.random.rand(*text.shape) < 1- self.space_count / self.punct_count ))
        mask |=  ((text == comma) & (np.random.rand(*text.shape) < 1- self.comma_count / self.punct_count )) 
        mask |=  (((text == period) | (text == chinese_period)) & (np.random.rand(*text.shape) < 1- self.p_count / self.punct_count))
        mask |=  ((text == question) & (np.random.rand(*text.shape) < 1- self.q_count / self.punct_count))
        mask &=  (targets != -1).type(torch.uint8) # corner case due to data inconsistencies

        mask = mask.bool()
        text[mask] = mask_token_number
        return text
    
    def full_mask_out(self,text,targets):
        """
        mask every single punctuation mark
        """
        period = self._preprocessor._tokenizer.encode(".",add_special_tokens=False)[1]
        chinese_period = self._preprocessor._tokenizer.encode("｡",add_special_tokens=False)[1]
        comma = self._preprocessor._tokenizer.encode(",",add_special_tokens=False)[1]
        question = self._preprocessor._tokenizer.encode("?",add_special_tokens=False)[0]
        mask_token_number = self._preprocessor._tokenizer.encode(self.mask_token,add_special_tokens=False)[0]
        self.mask_token_number = mask_token_number
        space_placeholder = 6
        
        values_mask = [period,comma,chinese_period,space_placeholder,question]
        mask =  sum(text == i for i in values_mask)
        mask &=  (targets != -1).type(torch.uint8)
        mask = mask.bool()
        
        text[mask] = mask_token_number
        return text        

    def validate(self, printer_counter):    
        # Valid loop
        self.model.eval()
        valid_loss = 0
        all_valid_preds = []
        all_valid_targets = []
        for data in tqdm(self.valid_loader):
            
            text, targets = data
            text = self.full_mask_out(text,targets)
            
            with torch.no_grad():
                preds, _ = self.model(text.to(self.device))
            
            mask_tokens = torch.tensor(text == self.mask_token_number).bool()
            predictions = preds[mask_tokens.unsqueeze(2).repeat((1,1,self._config.model.num_classes)).to(self.device)]
            targets = targets[mask_tokens]
            all_valid_targets.append(targets)
            
            loss = self.criterion(predictions.view(-1, self._config.model.num_classes), targets.to(self.device).view(-1))
            valid_loss += loss.mean().item()

            all_valid_preds.append(predictions.detach().cpu().numpy())

        valid_loss /= len(self.valid_loader)
        all_valid_preds = np.concatenate(all_valid_preds)
        all_valid_targets = np.concatenate(all_valid_targets)
        
        metrics = get_eval_metrics(all_valid_targets, all_valid_preds, self._config) # changed from self.all_valid_target
        metrics["loss"] = valid_loss

        print_metrics(printer_counter, metrics, self.summary_writer, 'valid',
                      model_name=self._config.model.name)
        return metrics
    
    def print_data(text, targets):
        textstring = []
        for i,j in zip(text[0], targets):
            if i == 250001:
                pass ## TODO
                
                
        
    def train(self):
        printer_counter = 0
        torch.autograd.set_detect_anomaly(True)
        if self._config.debug.break_train_loop:
            metrics = self.validate(printer_counter)
            return 0

        for epoch_num in range(self._config.trainer.num_epochs):
            log.info(f"Epoch #{self.epochs+epoch_num}")
            
            # Train loop
            self.model.train()
            pbar = tqdm(self.train_loader)
            counter = 0
            for data in pbar:
                
                counter+=1
                self.optimizer.zero_grad()

                text, targets = data
                
                starttime = time()
                text = self.full_mask_out(text,targets)
                endtime = time()
                wandb.log({'mask_time': endtime-starttime})
                if not (counter % 100):
                    print(text.shape)
                    print(self._preprocessor._tokenizer.decode(text[0]))
                    
                preds, binary_preds = self.model(text.to(self.device))

                # preds = preds[:, self._config.trainer.clip_seq: -self._config.trainer.clip_seq, :]
                # targets = targets[:, self._config.trainer.clip_seq:-self._config.trainer.clip_seq]

                # Mask some "empty" targets
                # mask = ((targets == 0) & (np.random.rand(*targets.shape) < .1)) | (targets > 0)
                # mask = mask.to(self.device)

                # Do not predict output after tokens which are not the end of a word
                starttime = time()
                mask_tokens = torch.tensor(text == self.mask_token_number).to(self.device).bool()
                
                targets = targets[mask_tokens] 
                
                predictions = preds[mask_tokens.unsqueeze(2).repeat((1,1,self._config.model.num_classes)).to(self.device)]   ## REQUIRES GRAD = True, out-of-place operation
                endtime = time()
                wandb.log({'pred_time': endtime-starttime})
                losses = self.criterion(predictions.reshape(-1, self._config.model.num_classes),
                                   targets.to(self.device).reshape(-1))
                loss = losses.mean()
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self._config.trainer.grad_clip)

                self.optimizer.step()
                self.sched.step()

                loss = loss.item()

                grads = get_total_grad_norm(self.model.parameters())
                pbar.set_postfix({"loss": loss, "grads": grads})

                print_metrics(printer_counter,
                              {"loss": loss, "grads": grads},
                              self.summary_writer, 'train',
                              model_name=self._config.model.name)
                printer_counter += 1

                wandb.log({'loss': loss, 'step': printer_counter})
            
                
                if self._config.model.save_model and (self._config.trainer.save_n_steps > 0) and not (counter % self._config.trainer.save_n_steps):
                    metrics = self.validate(printer_counter)
                    wandb.log({'valid_loss': metrics['loss'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f_score': metrics['f_score'],
                                'step': printer_counter})
                    save(self.model, self.optimizer, (self.epochs+epoch_num)+1+counter/10000, metrics, self._config)
                    print("saved model")
                

            metrics = self.validate(printer_counter)
            # Save model every epoch
            if self._config.model.save_model:
                save(self.model, self.optimizer, self.epochs+epoch_num+1, metrics, self._config)
                print("saved model")