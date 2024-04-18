from collections import OrderedDict

def populate_xtr_params(model, xtr_params):

    updated_model_states = []
    for key,value in model.state_dict().items():
        if key.startswith("xtr_layer"):
            mkey = key.split('.', 1)[1]
            updated_model_states.append((key, xtr_params[mkey]))
        else:
            updated_model_states.append((key, value))

    updated_model_states = OrderedDict(updated_model_states)

    model.load_state_dict(updated_model_states)
