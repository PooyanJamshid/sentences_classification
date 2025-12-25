import pandas as pd
from torch.utils.data import DataLoader
from function import build_dataset
from logger_config import get_logger

#/Users/pooyanjamshid/Desktop/vscode/class_sent/data

#read csv file
train_csv=pd.read_csv('../../data/Train.csv')#train data
valid_csv=pd.read_csv('../../data/Valid.csv')#valid data
test_csv=pd.read_csv('../../data/Test.csv')#test data

#build dataset
data_train=build_dataset(train_csv[:1000])
data_valid=build_dataset(valid_csv[:100])
data_test=build_dataset(test_csv[:100])

#build dataloader
train_loader=DataLoader(data_train,batch_size=10,shuffle=True)
valid_loader=DataLoader(data_valid,batch_size=10,shuffle=True)
test_loader=DataLoader(data_test,batch_size=10,shuffle=True)

logger = get_logger(__name__)
def preprocess():
    logger.info("Data Preprocessing Completed.")




