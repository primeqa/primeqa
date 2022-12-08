import grpc
import IPython
import json
import sys
from primeqa.services.grpc_server.grpc_generated.reader_pb2 import (
    GetReadersRequest, GetAnswersRequest, Contexts
    )
from primeqa.services.grpc_server.grpc_generated.reader_pb2_grpc import ReaderStub

with grpc.insecure_channel(f'localhost:50051') as channel:
        stub=ReaderStub(channel)
        rq=GetReadersRequest()
        #print(stub.GetReaders(rq))
        readers = stub.GetReaders(rq)

        reader=readers.readers[2]
        contexts0 = Contexts(texts=['the time is now', 'it was the best of times, it was the worst of times'])
        queries=['What time is it?', 'Is it now']
        contexts0 = Contexts(texts=['the time is now', 'it was the best of times, it was the worst of times'])
        contexts1 = Contexts(texts=['the time is now', 'it was the best of times, it was the worst of times', 'once upon a time'])
        arq=GetAnswersRequest(reader=reader, 
            queries=queries, 
            contexts=[contexts0, contexts1] 
        )
        print(stub.GetAnswers(arq))
