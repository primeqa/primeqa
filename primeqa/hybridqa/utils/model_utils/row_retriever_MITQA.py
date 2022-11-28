from torch.utils.data import WeightedRandomSampler
import argparse
from primeqa.hybridqa.processors.dataset import TableQADatasetQRSconcat
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer, 
                        BertModel, get_linear_schedule_with_warmup, 
                        squad_convert_examples_to_features)
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
from primeqa.hybridqa.models.table_encoder import RowClassifierSC
from primeqa.hybridqa.utils.partial_label_utils import partial_label_data_loader, pl_min_group_loss
from primeqa.hybridqa.utils.io_utils import read_data
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel,AdamW,get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import LongformerModel, LongformerTokenizer
from collections import OrderedDict
import json
import os



# Used during test if you want to test the model trained on multiple gpus on a single gpu.
def clean_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Training:
    def __init__(self,model,data_loader,batch_label_matrix,validator,save_every_niter,
                 save_model_path,optimizer,scheduler,device,criterion):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.batch_label_matrix = batch_label_matrix
        self.validator = validator
        self.save_every_niter = save_every_niter
        self.save_model_path = save_model_path

    def train(self, best_accuracy=0):
        self.model.train()
        losses =0.0
        correct_predictions = 0
        predictions_labels = []
        true_labels = []
        print("length of data loader",len(self.data_loader))
        for idx, (q_r_input,labels) in enumerate(tqdm(self.data_loader,total = len(self.data_loader),position=0, leave=True)):
            true_labels += labels.numpy().flatten().tolist()
            labels = labels.to(self.device)
            q_r_input = {k:v.type(torch.long).to(self.device) for k,v in q_r_input.items()}
            outputs = self.model(q_r_input,labels)
            label_matrix = self.batch_label_matrix[idx].to(self.device)
            #logits = outputs.logits
            logits =outputs.logits

            # Min group loss: 
            loss = pl_min_group_loss(logits, labels, label_matrix, self.criterion)
            
            losses+=loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            rounded_preds = torch.argmax(logits,axis=1)
            predictions_labels += rounded_preds.detach().cpu().numpy().tolist()
            if idx+1 % self.save_every_niter == 0:
                validation_labels,validation_predictions,validation_loss = self.validator.evaluate()
                validation_accuracy = accuracy_score(validation_labels,validation_predictions)
                print(f'Validation loss for iteration {idx+1}: {validation_loss}')

                if validation_accuracy > best_accuracy:
                    torch.save(self.model.state_dict(),self.save_model_path)
                    best_accuracy = validation_accuracy

        avg_epoch_loss = losses / len(self.data_loader)
        return true_labels, predictions_labels,avg_epoch_loss

class Validation:
    def __init__(self,model,data_loader,device,criterion):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion  = criterion
    def evaluate(self):
        self.model.eval()
        losses = 0
        correct_predictions = 0
        predictions_labels = []
        true_labels = []
        for q_r_input,labels in tqdm(self.data_loader,total = len(self.data_loader),position=0, leave=True):
            true_labels += labels.numpy().flatten().tolist()
            labels = labels.to(self.device)
            q_r_input = {k:v.type(torch.long).to(self.device) for k,v in q_r_input.items()}
            with torch.no_grad():
                outputs = self.model(q_r_input,labels)
                
                logits =outputs.logits
                loss = outputs.loss.mean()
                losses+=loss.item()
                rounded_preds = torch.argmax(logits,axis=1)
                predictions_labels += rounded_preds.detach().cpu().numpy().tolist()
        avg_epoch_loss = losses / len(self.data_loader)
        return true_labels, predictions_labels,avg_epoch_loss
class RowRetriever():
    def __init__(self,hqa_args,t_args):
        self.hqa_args = hqa_args
        self.t_args = t_args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RowClassifierSC(self.t_args.rr_model_name)

    def predict(self,processed_test_data):
 
        if self.hqa_args.test:
            state_dict = torch.load(self.t_args.row_retriever_model_name_path)
            state_dict = clean_model_state_dict(state_dict)
            # model = RowClassifierSC()
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(state_dict,strict=True)
            
            print("Model Loaded")
            self.model.to(self.device)
            self.model.eval()
            predictions_labels = []
            scores_list = []
            question_ids_list = []
            bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            #bert_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
            if processed_test_data is not None:
                #test_data = read_data(test_data_path)
                
                test_dataset = TableQADatasetQRSconcat(processed_test_data,512,bert_tokenizer,use_st_out=True,ret_labels=False)
                test_data_loader = DataLoader(test_dataset, batch_size=self.t_args.per_device_eval_batch_size_rr, batch_sampler=None, 
                                                num_workers=1, shuffle=False, pin_memory=True)
            for question_ids,q_r_input in tqdm(test_data_loader,total = len(test_data_loader),position=0, leave=True):
                q_r_input = {k:v.type(torch.long).to(self.device) for k,v in q_r_input.items()}
                
                with torch.no_grad():
                    outputs = self.model(q_r_input)
                    probs = outputs.logits
                    scores = probs[:,1]
                    scores = scores.detach().cpu().numpy().tolist() 
                    scores_list+=scores
                    question_ids_list+=question_ids
            
            q_id_scores_list = {}
            prev_qid = ""
            for q_id, score in zip(question_ids_list,scores_list):
                if q_id==prev_qid:
                    q_id_scores_list[q_id].append(score)
                else:
                    q_id_scores_list[q_id] = [score]
                    prev_qid=q_id
                
            json.dump(q_id_scores_list,open(os.path.join(self.hqa_args.data_path_root,"row_ret_scores.json"),"w"))
        return q_id_scores_list

    def train(self,train_data_processed,dev_data_processed):
        # if self.t_args.row_retriever_model_name_path != '':
        #     state_dict = torch.load(self.t_args.row_retriever_model_name_path)
        #     self.model.load_state_dict(state_dict,strict=True)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        bert_tokenizer = BertTokenizer.from_pretrained(self.t_args.rr_model_name)
        print('Model Loaded')

        #Load train and dev data from raw files
        #train_data = read_data(train_data_path)
        #dev_data = read_data(dev_data_path)
        train_data = train_data_processed
        dev_data = dev_data_processed

        use_st_out = False
        if use_st_out:
            use_st_out=True

        train_dataset = TableQADatasetQRSconcat(train_data, 512,bert_tokenizer, use_st_out=use_st_out,ret_labels=True)
        dev_dataset = TableQADatasetQRSconcat(dev_data, 512,bert_tokenizer, use_st_out=use_st_out,ret_labels=True)
        dev_data_loader, _ = partial_label_data_loader(dev_data,
                                                    tokenized_data = dev_dataset,
                                                    batch_size=self.t_args.per_device_eval_batch_size_rr)

        print('Partial label dataset-loaders created!')

        #loss function
        criterion = nn.CrossEntropyLoss(reduce=False) # using reduce=False to get loss per instance
        total_steps = len(train_data)*self.t_args.num_train_epochs_rr

        #Optimizer to update model parameters
        optimizer = AdamW(self.model.parameters(),lr = 5e-5,eps = 1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0.1*total_steps,
                                                    num_training_steps= total_steps)

        validator = Validation(data_loader = dev_data_loader,
                        model = self.model,
                        device = self.device,criterion=criterion
                        )

        pos_frac_list = self.t_args.pos_frac_per_epoch
        group_frac_list = self.t_args.group_frac_per_epoch
        
        # if pos_frac and grup_frac list aren't filled then put default values
        if pos_frac_list == []:
            pos_frac_list = [0.0001] * self.t_args.num_train_epochs_rr
        elif len(pos_frac_list) < self.t_args.num_train_epochs_rr:
            pos_frac_list += [pos_frac_list[-1]]* self.t_args.num_train_epochs_rr - len(pos_frac_list)

        if group_frac_list == []:
            group_frac_list = [1.0] * self.t_args.num_train_epochs_rr
        elif len(group_frac_list) < self.t_args.num_train_epochs_rr:
            group_frac_list += [group_frac_list[-1]]* (self.t_args.num_train_epochs_rr - len(group_frac_list))


        pos_fraction = pos_frac_list[0]
        group_fraction = group_frac_list[0] 
        train_data_loader, batch_label_matrix = partial_label_data_loader(train_data,
                                                    tokenized_data = train_dataset,
                                                    batch_size=self.t_args.per_device_train_batch_size_rr, 
                                                    pos_fraction = pos_fraction,
                                                    group_fraction = group_fraction)
        trainer = Training(data_loader = train_data_loader,
                    batch_label_matrix = batch_label_matrix,
                    model = self.model,
                    optimizer = optimizer,
                    validator = validator,
                    save_every_niter = self.t_args.save_every_niter_rr,
                    save_model_path = self.t_args.save_model_path_rr,
                    scheduler = scheduler,
                    device = self.device,criterion=criterion)

        best_accuracy = 0.0
        all_loss = {'train_loss':[], 'val_loss':[]}
        all_acc = {'train_acc':[], 'val_acc':[]}
        for epoch in tqdm(range(self.t_args.num_train_epochs_rr),position=0, leave=True):
            # data loader updated if pos_frac and group_frac is different from last epoch
            if pos_frac_list[epoch] != pos_fraction or group_frac_list[epoch] != group_fraction:
                pos_fraction = pos_frac_list[epoch]
                group_fraction = group_frac_list[epoch]
                train_data_loader, batch_label_matrix = partial_label_data_loader(train_data,
                                                        tokenized_data = train_dataset,
                                                        batch_size=self.t_args.per_device_train_batch_size_rr, 
                                                        pos_fraction = pos_fraction,
                                                        group_fraction = group_fraction)
                trainer = Training(data_loader = train_data_loader,
                        batch_label_matrix = batch_label_matrix,
                        model = self.model,
                        optimizer = optimizer,
                        validator = validator,
                        save_every_niter = self.t_args.save_every_niter_rr,
                        save_model_path = self.t_args.save_model_path_rr,
                        scheduler = scheduler,
                        device = self.device,criterion=criterion)

            print('Training now ')
            train_labels,train_predictions,training_loss = trainer.train(best_accuracy)
            
            training_accuracy = accuracy_score(train_labels,train_predictions)
            print(f'Training loss for epoch {epoch+1}: {training_loss}' )
            print(f'Training accuracy for epoch {epoch+1}: {training_accuracy}')

            validation_labels,validation_predictions,validation_loss = validator.evaluate()
            validation_accuracy = accuracy_score(validation_labels,validation_predictions)
            print(f'Validation loss for epoch {epoch+1}: {validation_loss}')

            if validation_accuracy > best_accuracy:
                torch.save(self.model.state_dict(),self.t_args.save_model_path_rr)
                best_accuracy = validation_accuracy

            all_loss['train_loss'].append(training_loss)
            all_loss['val_loss'].append(validation_loss)
            all_acc['train_acc'].append(training_accuracy)
            all_acc['val_acc'].append(validation_accuracy)

        print(f'Best Accuracy achieved: {best_accuracy}')

if __name__ == "__main__":
    main()