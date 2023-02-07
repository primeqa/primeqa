# How to add your own API
Let us explain through the example of TableReader API which is aimed to answer natural language questions over tabular data. 
## As Rest APIs
1. Define the I/O dataclasses: Define the input class and output class for your target API in [data_models.py](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/data_models.py).
Example: For TableReader , we introduced the input class [getTableAnswerRequest](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/data_models.py#L36) and output class [TableAnswer](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/data_models.py#L50)
2. Implement your API in [server.py] with the following steps. 
    a. define the API endpoint. Ex: [get\_table\_answer](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/server.py#L95) 
    b. Set the API's response\_model as the output clas you defined in data\_models. Ex: [TableAnswer](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/server.py#L97) for TableReader 
    c. Reuse/define a tag for your API- it reflects the high level functionaity the API is associated with. Ex: [TableReader](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/server.py#L98)
    d. Implement the API functionality which takes the input class defined in data\_models as input and uses modules from PrimeQA core. Ex: [get\_answer\_from\_tables](https://github.com/primeqa/primeqa/blob/table_reader_service/primeqa/services/rest_server/server.py#L100) takes getTableAnswerRequest as input , uses PrimeQA core module TapexReader from /primeqa/core/tableqa/tapex/tapex\_component.py to produce TableAnswer (defined in data\_models). 

## As GRPC Calls
TBD