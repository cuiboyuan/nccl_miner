import logging

# def setup_logger():
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     handler = logging.StreamHandler()  # You can also add FileHandler, etc.
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     return logger

# # Initialize the logger in the main file
# logger = setup_logger()


def logd(msg):
    logging.debug(msg, stacklevel=2)

def loge(msg):
    logging.error(msg, stacklevel=2)

def logi(msg):
    logging.info(msg, stacklevel=2)
