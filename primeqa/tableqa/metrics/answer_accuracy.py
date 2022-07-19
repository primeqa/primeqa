import logging

logger = logging.getLogger(__name__)

def compute_denotation_accuracy(dataset,predictions):

    logger.info(dataset.shape,predictions.predictions.shape)