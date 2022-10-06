from typing import List

from primeqa.pipelines.components.base import ReaderComponent


class ExtractiveReaderPipeline:
    def __init__(self, reader: ReaderComponent) -> None:
        self.reader = reader

    def run(self, input_texts: List[str], context: List[List[str]]):
        # Step 1: Load component
        self.reader.load()

        # Step 2: Run component
        return self.reader.apply(input_texts=input_texts, context=context)
