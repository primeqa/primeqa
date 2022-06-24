#from transformers import TapasConfig, TapasForQuestionAnswering, AdamW
from primeqa.tableqa.models.tableqa_model import TableQAModel
from transformers import TapasConfig


def main():
    print("Main function for training and testing tapas based tableqa")
    
    model = TableQAModel("google/tapas-base-finetuned-wtq")
    data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio",
                       "George Clooney"], "Number of movies": ["87", "53", "69"]}
    queries = ["What is the name of the first actor?",
               "How many movies has George Clooney played in?",
               "What is the total number of movies?", ]
    print(model.predict_from_dict(data,queries))

    config = TapasConfig(
    num_aggregation_labels=4,
    use_answer_as_supervision=True,
    answer_loss_cutoff=0.664694,
    cell_selection_preference=0.207951,
    huber_loss_delta=0.121194,
    init_cell_selection_weights_to_zero=True,
    select_one_column=True,
    allow_empty_column_selection=False,
    temperature=0.0352513,
    )
if __name__ == '__main__':
       main()


# # this is the default WTQ configuration
# config = TapasConfig(
#            num_aggregation_labels = 4,
#            use_answer_as_supervision = True,
#            answer_loss_cutoff = 0.664694,
#            cell_selection_preference = 0.207951,
#            huber_loss_delta = 0.121194,
#            init_cell_selection_weights_to_zero = True,
#            select_one_column = True,
#            allow_empty_column_selection = False,
#            temperature = 0.0352513,
# )
# model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

# optimizer = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(2):  # loop over the dataset multiple times
#    for idx, batch in enumerate(train_dataloader):
#         # get the inputs;
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         token_type_ids = batch["token_type_ids"]
#         labels = batch["labels"]
#         numeric_values = batch["numeric_values"]
#         numeric_values_scale = batch["numeric_values_scale"]
#         float_answer = batch["float_answer"]

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                        labels=labels, numeric_values=numeric_values, numeric_values_scale=numeric_values_scale,
#                        float_answer=float_answer)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()