import logging
import torch
import os
from transformers import PreTrainedTokenizer
from primeqa.util.transformers_utils.hypers_base import HypersBase

logger = logging.getLogger(__name__)


def save_transformer(hypers: HypersBase, model, tokenizer, *, save_dir=None):
    if hypers.global_rank == 0:
        if save_dir is None:
            save_dir = hypers.output_dir
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, "training_args.bin"))
        model_to_save.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)


def load_tokenizer(hypers: HypersBase, tokenizer_class, additional_special_tokens=()):
    if len(additional_special_tokens) == 0 or len(additional_special_tokens[0]) == 0:
        additional_special_tokens = None
    if additional_special_tokens is not None:
        tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
            hypers.tokenizer_name if hypers.tokenizer_name else hypers.model_name_or_path,
            do_lower_case=hypers.do_lower_case,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None,
            additional_special_tokens=additional_special_tokens
        )
    else:
        tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
            hypers.tokenizer_name if hypers.tokenizer_name else hypers.model_name_or_path,
            do_lower_case=hypers.do_lower_case,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None
        )
    return tokenizer


def load_pretrained(hypers: HypersBase, config_class, model_class, tokenizer_class,
                    additional_special_tokens=(), **extra_model_args):
    # Load pretrained model and tokenizer
    if hypers.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = config_class.from_pretrained(
        hypers.config_name if hypers.config_name else hypers.model_name_or_path,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
        **extra_model_args
    )
    if len(additional_special_tokens) == 0 or len(additional_special_tokens[0]) == 0:
        additional_special_tokens = None
    if additional_special_tokens is not None:
        tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
            hypers.tokenizer_name if hypers.tokenizer_name else hypers.model_name_or_path,
            do_lower_case=hypers.do_lower_case,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None,
            additional_special_tokens=additional_special_tokens
        )
    else:
        tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
            hypers.tokenizer_name if hypers.tokenizer_name else hypers.model_name_or_path,
            do_lower_case=hypers.do_lower_case,
            cache_dir=hypers.cache_dir if hypers.cache_dir else None
        )
    model = model_class.from_pretrained(
        hypers.model_name_or_path,
        from_tf=bool(".ckpt" in hypers.model_name_or_path),
        config=config,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
    )
    if hypers.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    if additional_special_tokens is not None:
        # do it when we load and again here
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        model.resize_token_embeddings(len(tokenizer))
    model.to(hypers.device)
    return model, tokenizer


def save_extended_model(hypers: HypersBase, model, tokenizer: PreTrainedTokenizer, *, save_dir=None):
    if hypers.global_rank == 0:
        if save_dir is None:
            save_dir = hypers.output_dir
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model to %s", save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, "training_args.bin"))
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.bin"))
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)


def load_only_extended_model(args: HypersBase, extended_model_class, saved_dir: str, *, strict=True):
    logger.info(f'loading model from {saved_dir}')
    if strict:
        hypers = torch.load(os.path.join(saved_dir, "training_args.bin"), map_location='cpu')
        hypers.device = args.device
    else:
        hypers = args
    model = extended_model_class(hypers)
    model_state_dict = torch.load(os.path.join(saved_dir, "model.bin"), map_location='cpu')
    model.load_state_dict(model_state_dict, strict=strict)
    model.to(args.device)
    return model, hypers
