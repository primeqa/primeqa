from typing import List
import logging
import time

from primeqa.pipelines.components.base import ReaderComponent
from primeqa.pipelines.base import ReaderPipeline


class ExtractiveReaderPipeline(ReaderPipeline):
    def __init__(
        self,
        reader: ReaderComponent,
        logger: logging.Logger = logging.getLogger("ExtractiveReaderPipeline"),
    ) -> None:
        self._logger = logger
        self.reader = reader

        # Default variables
        self.pipeline_id = self.__class__.__name__
        self.pipeline_name = "Extractive Reader"
        self.pipeline_description = ""
        self.pipeline_type = ReaderPipeline.__name__

    def load(self, *args, **kwargs):
        start_t = time.time()
        self.reader.load(*args, **kwargs)

        self._logger.info(
            "%s pipeline - loading took %s seconds",
            self.pipeline_name,
            time.time() - start_t,
        )

    def run(self, input_texts: List[str], context: List[List[str]], **kwargs):
        return self.reader.apply(input_texts=input_texts, context=context)
