import grpc

from primeqa.services.grpc_server.grpc_generated.reader_pb2 import (
    GetReadersRequest,
    GetAnswersRequest,
    Contexts
)
from primeqa.services.grpc_server.grpc_generated.reader_pb2_grpc import ReadingServiceStub as ReaderStub

#server=f'localhost:50051'
server=f'mnlp-qa-dev-2.sl.cloud9.ibm.com:50053'
with grpc.insecure_channel(server) as channel:
    stub=ReaderStub(channel)
    rq=GetReadersRequest()
    readers = stub.GetReaders(rq)
    reader=readers.readers[1]
    print(readers)
    print(reader.reader_id)

    import IPython
    IPython.embed()


# example of setting a parameter
# I think this goes through post_init, not load
# readers.readers[1].parameters[-1].value.string_value='factoid boolean'

    q0='has big ben got a crack in it?'
    q1='is new york considered a new england state'
    queries=[q1]
    t0="Since the tower was not yet finished, the bell was mounted in New Palace Yard but, during testing it cracked beyond repair and a replacement had to be made. The bell was recast on 10 April 1858 at the Whitechapel Bell Foundry as a 131⁄2 ton (13.76-tonne) bell. The second bell was transported from the foundry to the tower on a trolley drawn by sixteen horses, with crowds cheering its progress; it was then pulled 200 ft (61.0 m) up to the Clock Tower's belfry, a feat that took 18 hours. It is 7 feet 6 inches (2.29 m) tall and 9 feet (2.74 m) diameter. This new bell first chimed in July 1859; in September it too cracked under the hammer. According to the foundry's manager, George Mears, the horologist Denison had used a hammer more than twice the maximum weight specified. For three years Big Ben was taken out of commission and the hours were struck on the lowest of the quarter bells until it was repaired. To make the repair, a square piece of metal was chipped out from the rim around the crack, and the bell given an eighth of a turn so the new hammer struck in a different place. Big Ben has chimed with a slightly different tone ever since, and is still in use today with the crack unrepaired. Big Ben was the largest bell in the British Isles until ``Great Paul'', a 163⁄4 ton (17 tonne) bell currently hung in St Paul's Cathedral, was cast in 1881."
    t1="New England is a geographical region comprising six states of the northeastern United States: Maine, Vermont, New Hampshire, Massachusetts, Rhode Island, and Connecticut. It is bordered by the state of New York to the west and by the Canadian provinces of New Brunswick and Quebec to the northeast and north, respectively. The Atlantic Ocean is to the east and southeast, and Long Island Sound is to the south. Boston is New England's largest city as well as the capital of Massachusetts. The largest metropolitan area is Greater Boston, which also includes Worcester, Massachusetts (the second-largest city in New England), Manchester (the largest city in New Hampshire), and Providence (the capital and largest city of Rhode Island), with nearly a third of the entire region's population."
    contexts0 = Contexts(texts=[t1])
    #contexts1 = Contexts(texts=[t0,t1])
    arq=GetAnswersRequest(reader=reader, 
        queries=queries, 
        contexts=[contexts0]
    )
    print(stub.GetAnswers(arq))
