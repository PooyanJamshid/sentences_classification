import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from logger_config import get_logger
from torch.utils.data import Dataset
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



#device for train model and preprocess ,....

device=torch.device('cuda' if torch.cuda.is_available() else'cpu')


# class for structure model
class model(torch.nn.Module):
    def __init__(self,model,tooken,out_shape,n_class):

        super().__init__()
        self.model=model.to(device)
        self.token=tooken.to(device)
        #classfier
        self.classifier=nn.Sequential(
            nn.Linear(out_shape,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,74),
            nn.ReLU(),
            nn.Linear(74,n_class)
            )

    def forward(self,text):
        #tookenize model
        inputs=self.token(
            list(text),
            return_tensors='pt',
            padding=True, truncation=True)
        
        #inputs=inputs.to(device)
        outputs=self.model(input_ids=inputs['input_ids'],
                           attention_mask=inputs['attention_mask'])
        
        #get cls token
        cls=outputs.last_hidden_state[:,0,:]
        idx=self.classifier(cls)
        return idx
    
#function for train model

def train_model(model,train_loader,epochs,loss_function,jump,lr):
    loss_fn=loss_function
    #Adam
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    train_list_loss=[]
    train_list_cross=[]
    #started for training
    model.train()

    for epoch in range(epochs):  
        total_loss=0
        for labels,texts in train_loader:
            labels=labels.to(device)
            # make all gradien prepaid for training 
            optimizer.zero_grad()

            outputs=model(texts)
            #my loss function is croos entropy so i have to get vector of outputs and ->
            # don't use arg max
            loss=loss_fn(outputs,labels)

            loss.backward()

            optimizer.step()

            total_loss+=loss.item()

        avg_loss=total_loss/len(train_loader)
        #this loss is for show training process not for training model
        predict=model(texts)
        loss_cross=loss_fn(predict,labels)

        #i deleted part of question abour save model

        train_list_loss.append(avg_loss)
        train_list_cross.append(loss_cross.item())

    return train_list_loss,train_list_cross

# load weight of  model in (sent_class.ipyb)
def weight_add_function(model, adress_weight, device='cpu'):
    checkpoint = torch.load(adress_weight, map_location=torch.device(device))
    new_state_dict = {
        k[len("model."):] if k.startswith("model.") else k: v
        for k, v in checkpoint.items()
    }
    model.load_state_dict(new_state_dict, strict=False)
    return model

# make data set 

class build_dataset(Dataset):
    def __init__(self,data):
        self.data=data
      

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        #get text and label

        text=str(self.data['text'][index])
        label = torch.tensor(self.data['label'][index], dtype=torch.long)
        


        return label,text


def test_model(model,loader,device):


  #stop dropout
  model.eval()
  true_labels=[]
  predicted_labels=[]
  #stop training 
  with torch.no_grad():
    for labels,texts in loader:
      labels=labels.to(device)
      #predict data by model
      outputs=model(texts)
      #get finally label(index)
      predicted=torch.argmax(outputs,1)
      true_labels.extend(labels.cpu().numpy())
      predicted_labels.extend(predicted.cpu().numpy())
      #because limited computational resources i dont test on all data test
      break
  
  f1_score_value=f1_score(true_labels,predicted_labels, average='weighted')
  accuracy=accuracy_score(true_labels,predicted_labels)
  precision=precision_score(true_labels,predicted_labels, average='weighted')
  recall=recall_score(true_labels,predicted_labels, average='weighted')
  report=classification_report(true_labels,predicted_labels)

  return((f"f1_score:{f1_score_value},accuracy:{accuracy},precision:{precision},'recall':{recall}"),report)
  
def function():
    logger = get_logger(__name__)

    logger.info('function is called')
