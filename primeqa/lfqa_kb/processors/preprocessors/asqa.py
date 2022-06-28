def preprocess_asqa_validation_function(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    inputs, targets = preprocess_asqa_batch(examples, data_args, mode="eval")

    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding=padding,
        truncation=True,
        # return_overflowing_tokens=True, # See this issue https://github.com/huggingface/transformers/issues/15398#issuecomment-1072714545
        return_offsets_mapping=True,
    )
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

    model_inputs["example_id"] = []

    for i in range(len(model_inputs["input_ids"])):
        model_inputs["example_id"].append(examples["id"][i])

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_asqa_function(examples, data_args, tokenizer, max_seq_length, max_answer_length, padding):
    inputs, targets = preprocess_asqa_batch(examples, data_args, mode="train")
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_asqa_batch(examples, data_args, mode="train"):
    def generate_input(_question):
        return " ".join(["question:", _question.lstrip()])
    questions = examples[data_args.question_column] # "ambiguous_question"
    answers = examples[data_args.answer_column] # "annotations"
    if mode == "train":
        inputs = []
        targets = []
        for idx, question in enumerate(questions):
            q = question
            answer_list = answers[idx]
            if len(answer_list) == 0:
                inputs.append(q)
                targets.append("")
            else:
                for answer_data in answer_list:
                    a = answer_data["long_answer"]
                    inputs.append(q)
                    targets.append(a)
    elif mode == "eval":
        inputs = [generate_input(question) for question in questions]
        targets = [answer[0]["long_answer"] if len(answer) > 0 else "" for answer in answers]

    else:
        raise ValueError("mode requires eval or train")
    return inputs, targets