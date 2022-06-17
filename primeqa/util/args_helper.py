from argparse import ArgumentParser
from enum import Enum
import logging

logger = logging.getLogger(__name__)


def fill_from_dict(defaults, a_dict):
    set_values = []
    for arg, val in a_dict.items():
        if val is None:
            continue
        set_values.append(arg)
        parts = arg.split('.')  # set nested objects with names like obj1.attr
        toset = defaults
        for part in parts[:-1]:
            toset = toset.__dict__[part]
        name = parts[-1]
        d = toset.__dict__[name]
        if isinstance(d, Enum):
            toset.__dict__[name] = type(d)[val]
        else:
            toset.__dict__[name] = val
    return set_values


def _nested_get(obj, arg):
    parts = arg.split('.')  # set nested objects with names like obj1.attr
    toget = obj
    for part in parts[:-1]:
        if not hasattr(toget, '__dict__') or part not in toget.__dict__:
            return None
        toget = toget.__dict__[part]
    return toget.__dict__[parts[-1]]


def name_value_list(obj, prefix=''):
    name_values = []
    for attr, value in obj.__dict__.items():
        # ignore members that start with '_'
        if attr.startswith('_'):
            continue
        if hasattr(value, '__dict__'):
            name_values.extend(name_value_list(value, prefix=prefix + attr + '.'))
        else:
            name_values.append((prefix+attr, value))
    return name_values


def fill_from_args(defaults):
    """
    Builds an argument parser, parses the arguments, updates and returns the object 'defaults'
    :param defaults: an object with fields to be filled from command line arguments
    :return:
    """
    parser = ArgumentParser()
    # if defaults has a __required_args__ we set those to be required on the command line
    required_args = []
    names_values = name_value_list(defaults)
    # print(names_values)
    names = set([n for n, v in names_values])
    if hasattr(defaults, '__required_args__'):
        required_args = defaults.__required_args__
        for reqarg in required_args:
            if reqarg.startswith('_'):
                raise ValueError(f'arguments should not start with an underscore ({reqarg})')
            if reqarg not in names:
                raise ValueError(f'argument "{reqarg}" is required, but not present in __init__')
    for attr, value in names_values:
        help_str = None
        # if it is a tuple, we assume the second is the help string
        # if type(value) is tuple and len(value) == 2 and type(value[1]) is str:
        #     help_str = value[1]
        #     value = value[0]

        # check if it is a type we can take on the command line
        if type(value) not in [str, int, float, bool] and not isinstance(value, Enum):
            raise ValueError(f'Error on {attr}: cannot have {type(value)} as argument')
        if type(value) is bool and value:
            raise ValueError(f'Error on {attr}: boolean arguments (flags) must be false by default')

        # also handle str to enum conversion
        t = type(value)
        if isinstance(value, Enum):
            t = str

        # don't pass defaults to argparse, just pass None we'll keep the default value if the arg value is None
        if t is bool:
            # support bool with store_true (required false by default)
            parser.add_argument('--'+attr, default=None, action='store_true', help=help_str)
        else:
            parser.add_argument('--'+attr, type=t, default=None, help=help_str, required=(attr in required_args))
    args = parser.parse_args()
    # now update the passed object with the arguments
    defaults.__passed_args__ = fill_from_dict(defaults, args.__dict__)
    # call _post_argparse() if the method is defined
    try:
        defaults._post_argparse()
    except AttributeError:
        pass
    return defaults
