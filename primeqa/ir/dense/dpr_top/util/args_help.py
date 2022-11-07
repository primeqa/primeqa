from argparse import ArgumentParser
from enum import Enum
import logging
import dataclasses
from typing import Optional, Dict, List, Union

logger = logging.getLogger(__name__)


def fill_from_dict(defaults, a_dict) -> List:
    set_values = []
    for arg, val in a_dict.items():
        if val is None:
            continue
        set_values.append(arg)
        parts = arg.split('.')  # set nested objects with names like obj1.attr
        toset = defaults
        # CONSIDER: use _dict_get (getattr or dict lookup)
        for part in parts[:-1]:
            toset = toset.__dict__[part]
        name = parts[-1]
        d = toset.__dict__[name]
        # CONSIDER: use _dict_set (setattr or dict assign)
        if isinstance(d, Enum):
            toset.__dict__[name] = type(d)[val]
        else:
            toset.__dict__[name] = val
    return set_values


def _get_dict(obj) -> Optional[Dict]:
    """
    get a dictionary from obj, works if obj is a dict, dataclass or has a __dict__ or __slots__
    :param obj:
    :return: a dict, writes to the dict will not necessarily update obj
    """
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, '__slots__'):
        return {slot: getattr(obj, slot) for slot in obj.__slots__}
    if hasattr(obj, '__dict__') and not isinstance(obj, Enum):
        return obj.__dict__
    # raise ValueError(f'cannot coerce {type(obj)} to dict')
    return None


def _nested_get(obj, arg: Union[str, List[str]]):
    if type(arg) == str:
        parts = arg.split('.')  # set nested objects with names like obj1.attr
    else:
        parts = arg
    toget = obj
    for part in parts[:-1]:
        toget_dict = _get_dict(toget)
        if toget_dict is None or part not in toget_dict:
             return None
        toget = toget[part]
    toget_dict = _get_dict(toget)
    if toget_dict is None or parts[-1] not in toget_dict:
        return None
    return toget_dict[parts[-1]]


def name_value_list(obj, prefix=''):
    """
    get the obj as a list of name, value pairs. nested attributes are separated with '.'
    :param obj:
    :param prefix:
    :return:
    """
    name_values = []
    obj = _get_dict(obj)
    for attr, value in obj.items():
        # ignore members that start with '_'
        if attr.startswith('_'):
            continue
        if _get_dict(value) is not None:
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
    if hasattr(defaults, '_required_args') or hasattr(defaults, '__required_args__'):
        names = set([n for n, v in names_values])
        required_args = defaults._required_args if hasattr(defaults, '_required_args') else defaults.__required_args__
        for reqarg in required_args:
            if reqarg.startswith('_'):
                raise ValueError(f'arguments should not start with an underscore ({reqarg})')
            if reqarg not in names:
                raise ValueError(f'argument "{reqarg}" is required, but not present in __init__')
    for attr, value in names_values:
        # CONSIDER: permit help strings like self._help = {'name': 'help string', ...}
        help_str = None

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
    # CONSIDER: should there be general arguments to set log level?
    # CONSIDER: switch __required_args__ to just _required_args and same for __passed_args__
    return defaults
