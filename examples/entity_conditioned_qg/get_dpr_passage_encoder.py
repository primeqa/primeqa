from dpr.models import init_biencoder_components
import argparse
from dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint
from dpr.options import set_encoder_params_from_state
import torch

def get_passage_encoder(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    _, encoder, _ = init_biencoder_components("hf_bert", args, inference_only=True)
    encoder = encoder.ctx_model
    encoder.eval()
    model_to_load = get_model_obj(encoder)
    print("Loading saved model state")
    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)
    model_to_save = get_model_obj(encoder)
    torch.save(encoder.state_dict(), args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--model_file', type=str, help="input model path")
    parser.add_argument('--output_path', type=str, help="output encoder path")
    parser.add_argument('--pretrained_model_cfg', type=str, help="output encoder path")
    parser.add_argument("--projection_dim", default=0, type=int,
                        help="Extra linear layer on top of standard bert/roberta encoder")
    args = parser.parse_args()

    get_passage_encoder(args)