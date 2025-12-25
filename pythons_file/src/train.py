from model import model_main_albert
from function import device,train_model
from hyperparameter import lr,jump,epochs,loss_function
from preprocess import train_loader
from logger_config import get_logger



logger = get_logger(__name__)

def train_start():
    logger.info("train started.")

#model have to be loaded here
finally_model=model_main_albert.to(device)

#finally_model=weight_add_function(finally_model)

#call fntion for training model
train_loss,cross_loss=train_model(finally_model,train_loader,epochs,loss_function,jump,lr)


logger = get_logger(__name__)
def train_ended():
    logger.info("train completed.")