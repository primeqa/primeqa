from transformers import Seq2SeqTrainer

class QGTrainer(Seq2SeqTrainer):
    """ The trainer class for QG. All related functionality should go to this class. """

    def __init__(self, *args, **kwargs):
        """ Custom intialization for the QGTrainer should be added here. """
        super().__init__(*args, **kwargs)