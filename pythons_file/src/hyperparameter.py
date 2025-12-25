from torch import nn
from logger_config import get_logger

batch_size=32
lr=2e-5
epochs=1
jump=1
loss_function=nn.CrossEntropyLoss()

logger = get_logger(__name__)

def get_hyperparameters(): 
    logger.info("Fetching hyperparameters for training.")