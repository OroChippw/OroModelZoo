import logging
from types import MethodType

def epochInfo(self , idx , loss , acc):
    self.info(f'{idx:d} epoch | loss : {loss:.8f} | acc : {acc:.4f}'.format(
        idx=idx , loss=loss , acc=acc
    ))

def get_logger():
    logger = logging.getLogger('')
    logger.epochInfo = MethodType(epochInfo , logger)

