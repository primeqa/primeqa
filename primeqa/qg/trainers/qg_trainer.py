from typing import Dict

import torch
from datasets import Dataset
from primeqa.qg.utils.data import find_answer_span
from tqdm import tqdm
from transformers import Seq2SeqTrainer


class QGTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        """The trainer class for QG. All related functionality should go to this class."""
        super().__init__(*args, **kwargs)


class GenTrainer(Seq2SeqTrainer):
    def __init__(self, max_gen_length: int, *args, **kwargs):
        """A `transformers.Trainer` which can be used for training, evaluation and generation of generation models yielding question-answer pairs (RC samples) given passages

        Args:
            max_gen_length (int): The maximum generation length. Note that this is computed per generation for multiple decoding steps.
        """
        super().__init__(*args, **kwargs)
        if max_gen_length is None:
            raise ValueError("`max_gen_length` cannot be None")
        self.max_gen_length = max_gen_length

        # set some token ids
        soq_token_id = self.tokenizer.encode("<q>", add_special_tokens=False)
        soa_token_id = self.tokenizer.encode("<a>", add_special_tokens=False)
        assert len(soq_token_id) == len(soa_token_id) == 1
        self.soq_token_id, self.soa_token_id = soq_token_id[0], soa_token_id[0]
        eoq_token_id = self.tokenizer.encode("</q>", add_special_tokens=False)
        eoa_token_id = self.tokenizer.encode("</a>", add_special_tokens=False)
        assert len(eoq_token_id) == len(eoa_token_id) == 1
        self.eoa_token_id = eoa_token_id[0]
        self.eoq_token_id = eoq_token_id[0]

    def _extract(self, token_ids, second: bool = False):
        def get_token_index(token_ids, start_index, token_id):
            return (token_ids[start_index:] == token_id).nonzero(as_tuple=True)[0]

        if second:
            # extract answer only (second step)
            soa_indices = get_token_index(token_ids, 0, self.soa_token_id)
            if len(soa_indices) == 0:
                # can't extract question
                return None, None, None, None
            answer_start_index = soa_indices[0] + 1
            eoa_indices = get_token_index(token_ids, answer_start_index, self.eoa_token_id)
            if len(eoa_indices) == 0:
                # can't extract answer
                return None, None, None, None
            answer_end_index = answer_start_index + eoa_indices[0]
            return None, None, answer_start_index, answer_end_index
        else:
            # extract question only (first step)
            soq_indices = get_token_index(token_ids, 0, self.soq_token_id)
            if len(soq_indices) == 0:
                # can't extract question
                return None, None, None, None
            question_start_index = soq_indices[0] + 1
            eoq_indices = get_token_index(token_ids, question_start_index, self.eoq_token_id)
            if len(eoq_indices) == 0:
                # can't extract question
                return None, None, None, None
            question_end_index = question_start_index + eoq_indices[0]
            return question_start_index, question_end_index, None, None

    def _compute_lm_score(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        offset: int,
        start_index: int,
        end_index: int,
    ):
        # this is using the model again similar to what they do in https://huggingface.co/transformers/perplexity.html to compute the perplexity
        with torch.no_grad():
            target_ids = output_ids.clone().unsqueeze(0)
            target_ids[:, : offset + start_index] = -100
            target_ids[:, offset + end_index :] = -100

            target_ids = target_ids[..., 1:]
            decoder_input_ids = output_ids.clone().unsqueeze(0)[..., :-1]
            outputs = self.model(input_ids, labels=target_ids, decoder_input_ids=decoder_input_ids)
            # outputs[0] is the average negative log likelihood per token
            score = -1.0 * outputs[0].cpu().item() * (end_index - start_index)
        return score

    def _generate_and_extract(self, sample: Dict):
        """Generate output token ids from inputs and extract question and answer with scores by computing their spans in the decoded sequence"""
        # prepare inputs
        inputs = {
            "input_ids": torch.tensor(sample["input_ids"], device=torch.device(self.args.device)).unsqueeze(0),
            "attention_mask": torch.tensor(sample["attention_mask"], device=torch.device(self.args.device)).unsqueeze(
                0
            ),
        }
        if "token_type_ids" in sample:
            inputs["token_type_ids"] = torch.tensor(
                sample["token_type_ids"], device=torch.device(self.args.device)
            ).unsqueeze(0)

        # set eos token
        eos_token_id = self.tokenizer.encode("</q>", add_special_tokens=False)
        # eos token for second decoding step
        eos_token_id_2 = self.tokenizer.encode("</a>", add_special_tokens=False)
        assert len(eos_token_id_2) == 1
        eos_token_id_2 = eos_token_id_2[0]
        # if correctly added to the dictionary then those tokens won't be split
        assert len(eos_token_id) == 1
        eos_token_id = eos_token_id[0]

        # set bos token
        # bos token for second decoding step
        bos_token_id_2 = self.tokenizer.encode("<a>", add_special_tokens=False)
        assert len(bos_token_id_2) == 1
        bos_token_id_2 = bos_token_id_2[0]

        bos_token_id = self.tokenizer.encode("<q>", add_special_tokens=False)

        # if correctly added to the dictionary then those tokens won't be split
        assert len(bos_token_id) == 1
        bos_token_id = bos_token_id[0]

        # parameters from Shakeri et al.
        assert inputs["input_ids"].size(-1) + self.max_gen_length <= self.tokenizer.model_max_length
        generation_output = self.model.generate(
            **inputs,
            max_length=self.max_gen_length
            if self.model.config.is_encoder_decoder
            else inputs["input_ids"].size(-1) + self.max_gen_length,
            max_new_tokens=None,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            num_return_sequences=10,
            num_beams=1,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=bos_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            forced_eos_token_id=eos_token_id
        )

        # scores are before softmax hence we apply it as well as log (for summing the scores later)
        # generation_scores = torch.stack(generation_output.scores, dim=0).log_softmax(
        #     dim=-1
        # )

        questions, answers, scores = [], [], []
        for generated_sequence in generation_output.sequences:
            gen_sequence = generated_sequence
            question_start, question_end, answer_start, answer_end = self._extract(gen_sequence)
            if question_start == question_end == None:
                continue
            # make sure that indices are on CPU
            question_start, question_end = int(question_start), int(question_end)

            question = self.tokenizer.decode(
                gen_sequence[question_start:question_end],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            ).strip()

            # question cannot be empty
            if question == "":
                continue

            # we have to do a second decoding step
            # prepare input
            inputs = {
                "input_ids": torch.cat(
                    (
                        torch.tensor(sample["input_ids"], device=torch.device(self.args.device)),
                        torch.tensor([self.soq_token_id], device=torch.device(self.args.device)),
                        gen_sequence[question_start:question_end],
                    ),
                    dim=0,
                ).unsqueeze(0)
            }

            # do greedy decoding for answer
            generation_output = self.model.generate(
                **inputs,
                max_length=self.max_gen_length,
                max_new_tokens=None,
                num_return_sequences=1,
                num_beams=10,
                do_sample=False,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=bos_token_id_2,
                bos_token_id=bos_token_id_2,
                eos_token_id=eos_token_id_2,
                forced_eos_token_id=eos_token_id_2
            )

            # get score (answer score is sufficient for qa2s)
            # generation_scores = torch.stack(
            #     generation_output.scores, dim=0
            # ).log_softmax(dim=-1)

            generated_sequence = generation_output.sequences[0]
            gen_sequence = generated_sequence
            # this time we're interested in the answer only
            _, _, answer_start, answer_end = self._extract(gen_sequence, second=True)

            # we have to extract both question and answer
            if answer_start == answer_end == None:
                continue
            # make sure that indices are on CPU
            answer_start, answer_end = int(answer_start), int(answer_end)

            answer = self.tokenizer.decode(
                gen_sequence[answer_start:answer_end],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            ).strip()

            # make sure that answer is not empty and appears in context
            if answer == "" or answer not in sample["context"]:
                continue

            answers.append(answer)
            questions.append(question)

            # extract LM scores for later use in filtering
            # only use answer score
            score = self._compute_lm_score(inputs["input_ids"], generated_sequence, 0, answer_start, answer_end)
            scores.append(score)
            assert scores

        return questions, answers, scores

    def predict(self, test_dataset: Dataset):
        """Generate question-answer pairs from passages (have to be preprocessed into features)

        Args:
            test_dataset (Dataset): The data used as input for generation

        Returns:
            List[Dict]: The generated RC samples
        """
        # dataloader = self.get_test_dataloader(test_dataset)
        self.model.eval()

        predictions = []

        for sample in tqdm(test_dataset):
            questions, answers, scores = self._generate_and_extract(sample)
            if not questions:
                # no generated sample could be extracted
                continue

            # answer was predicted together with question
            gen_answers = []
            for answer in answers:
                char_start = find_answer_span(sample["context"], answer)[0]
                assert char_start >= 0
                gen_answers.append({"answer_start": [char_start], "text": [answer]})
            answers = gen_answers

            predictions.append(
                {
                    "id": sample["id"],
                    "scores": scores,
                    "context": sample["context"],
                    "questions": questions,
                    "answers": answers,
                }
            )

        return predictions
