from model import model_main_albert
from function import test_model,device
from preprocess import test_loader
from logger_config import get_logger

logger = get_logger(__name__)

def test_start():
    logger.info("model test started")

#test start
metrics,report=test_model(model_main_albert,test_loader,device)

logger = get_logger(__name__)

def test_ended():
    logger.info("model test ended.")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Report: {report}")
    