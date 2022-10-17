from typing import List

from primeqa.pipelines.components.base import ReaderComponent, RetrieverComponent

class QAPipeline:
    def __init__(self, retriever: RetrieverComponent, reader: ReaderComponent) -> None:
        self.retriever = retriever
        self.reader = reader

    def run(self, input_texts: List[str]):
        # Step 1: Load retriever component
        self.retriever.load()

        # Step 2: Run retriever component
        hits = self.retriever.retrieve(input_texts=input_texts)
        