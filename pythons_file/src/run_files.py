import subprocess
from logger_config import get_logger
from train import train_ended,train_start
from preprocess import preprocess
from model import log_model
from hyperparameter import get_hyperparameters
from function import function
from test import test_ended,test_start


logger = get_logger("MAIN")

if __name__ == "__main__":
    logger.info("Program started")

    subprocess.run(['python','logger_config.py'])

    subprocess.run(['python','hyperparameter.py'])
    get_hyperparameters()
    subprocess.run(['python','function.py'])
    function()

    subprocess.run(['python','preprocess.py'])
    preprocess()

    subprocess.run(['python','model.py'])
    log_model()

    train_start()

    subprocess.run(['python','train.py'])
    train_ended()


    test_start()
    subprocess.run(['python','test.py'])
    test_ended()


    logger.info("Program finished")


#subprocess.run(['python','hyperparameter.py'])

#subprocess.run(['python','function.py'])

#subprocess.run(['python','logger_config.py'])

#subprocess.run(['python','preprocess.py'])

#subprocess.run(['python','model.py'])

#subprocess.run(['python','train.py'])

#subprocess.run(['python','plots.py'])

#subprocess.run(['python','test.py'])

