import logging

from datasets import Dataset, DatasetDict, load_dataset
from primeqa.qg.processors.passage_qg.qa2s_processor import QA2SProcessor
from primeqa.qg.processors.passage_qg.qg_processor import QGProcessor
from primeqa.qg.processors.table_qg.sql_processor import SqlProcessor

logger = logging.getLogger(__name__)


class QGDataLoader:
    def __init__(
        self,
        tokenizer,
        modality,
        input_max_len,
        target_max_len,
        gen_config="qg",
        dataset_name=None,
        dataset_config=None,
        dataset_split=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split

        if modality == "table":
            self.processor = SqlProcessor(tokenizer, input_max_len, target_max_len)
        elif modality == "passage":
            if gen_config == "qg":
                self.processor = QGProcessor(tokenizer, input_max_len, target_max_len)
            elif gen_config == "qa2s":
                self.processor = QA2SProcessor(tokenizer, input_max_len, target_max_len)

    def create(
        self,
        dataset: Dataset = None,
        dataset_split: str = "train",
        dataset_config: str = None,
    ) -> Dataset:
        if dataset is None:
            # load dataset first
            assert self.dataset_name is not None
            dataset_split = (
                dataset_split if dataset_split is not None else self.dataset_split
            )
            dataset_config = (
                dataset_config if dataset_config is not None else self.dataset_config
            )
            # select `secondary task` config for TyDi QA if none is given
            if self.dataset_name == "tydiqa" and dataset_config is None:
                dataset_config = "secondary_task"
                logger.info(
                    "Defaulting to config '{dataset_config}' for dataset {self.dataset_name}"
                )
            dataset = load_dataset(
                self.dataset_name, name=dataset_config, split=dataset_split
            )
            if isinstance(dataset, DatasetDict):
                raise ValueError(
                    "Loaded dataset is of type DatasetDict, did you choose a split?"
                )
        else:
            assert(dataset_split is not None)
            dataset = dataset[dataset_split]

        # NOTE works only if data has correct format
        return self.processor(dataset)
