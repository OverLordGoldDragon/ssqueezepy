# -*- coding: utf-8 -*-
"""
Contains `GDEFAULTS`, global defaults dictionary, set in `ssqueezepy.configs.ini`.

The .ini is parsed into a dict, then values are retrieved internally by functions
via `gdefaults()`, which sets default values if keyword arguments weren't set
to original functions (or were set to `None`).

E.g. calling `wavelets.morlet()`, the function has `mu=None` signature, so `mu`
will be drawn from `configs.ini`, unless calling like `wavelets.morlet(mu=1)`.
"""
import os
import inspect
import logging

logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
path = os.path.join(os.path.dirname(__file__), 'configs.ini')

try:
    import torch
    import cupy
except:
    torch, cupy = None, None


def gdefaults(module_and_obj=None, get_all=False, as_dict=None,
              default_order=False, **kw):
    """Fetches default arguments from `ssqueezepy/configs.ini` and fills them
    in `kw` where they're None (or always if `get_all=True`). See code comments.
    """
    if as_dict is None:
        as_dict = bool(get_all)

    if module_and_obj is None:
        stack = inspect.stack(0)  # `(0)` faster than `()`
        obj = stack[1][3]
        module = stack[1][1].split(os.path.sep)[-1].rstrip('.py')
    else:
        # may have e.g. `utils.common.obj`
        mos = module_and_obj.split('.')
        module, obj = '.'.join(mos[:-1]), mos[-1]

    # fetch latest
    GDEFAULTS = _get_gdefaults()

    # if `module` & `obj` are found in `GDEFAULTS`, proceed to write values
    # from `GDEFAULTS` onto `kw` if `kw`'s are `None`
    # if `get_all=True`, load values from `GDEFAULTS` even if they're not in
    # `kw`, but don't overwrite those that are in `kw`.
    # if `default_order=True`, will return `kw` with keys sorted as in
    # `configs.ini`, for e.g. plotting purposes
    if module not in GDEFAULTS:
        WARN(f"module {module} not found in GDEFAULTS (see configs.ini)")
    elif obj not in GDEFAULTS[module]:
        WARN(f"object {obj} not found in GDEFAULTS['{module}'] "
             "(see configs.ini)")
    else:
        DEFAULTS = GDEFAULTS[module][obj]
        for key, value in kw.items():
            if value is None:
                kw[key] = DEFAULTS.get(key, value)

        if get_all:
            for key, value in DEFAULTS.items():
                if key not in kw:
                    kw[key] = value
        if default_order:
            # first make a dict with correct order
            # then overwrite its values with `kw`'s, without changing order
            # if `kw` has keys that `ordered_kw` doesn't, they're inserted at end
            ordered_kw = {}
            for key, value in DEFAULTS.items():
                if key in kw:  # `get_all` already accounted for
                    ordered_kw[key] = value
            ordered_kw.update(**kw)
            kw = ordered_kw

    if as_dict:
        return kw
    return (kw.values() if len(kw) != 1 else
            list(kw.values())[0])


def _get_gdefaults():
    """Global defaults fetched from configs.ini."""
    def float_if_number(s):
        """If float works, so should int."""
        if isinstance(s, (bool, type(None))):
            return s
        try:
            return float(s)
        except ValueError:
            return s

    def process_special(s):
        return {
            'None':  None,
            'True':  True,
            'False': False,
        }.get(s, s)

    def process_value(value):
        value = value.strip('"').strip("'")
        return float_if_number(process_special(value))

    with open(path, 'r') as f:
        txt = f.read().split('\n')
        txt = txt[:txt.index('#### END')]
        txt = [line.strip(' ') for line in txt if line != '']

    GDEFAULTS = {}
    module, obj = '', ''
    for line in txt:
        if line.startswith('## '):
            module = line[3:]
            GDEFAULTS[module] = {}
        elif line.startswith('# '):
            obj = line[2:]
            GDEFAULTS[module][obj] = {}
        else:
            key, value = [s.strip(' ') for s in line.split('=')]
            GDEFAULTS[module][obj][key] = process_value(value)
    return GDEFAULTS


def IS_PARALLEL():
    """Returns False if 'SSQ_PARALLEL' environment flag was set to '0', or
    if `parallel` in `configs.ini` is set to `0`; former overrides latter.
    """
    not_par_env = (os.environ.get('SSQ_PARALLEL', '1') == '0')
    if not_par_env:
        return False

    not_par_config = (gdefaults('configs.IS_PARALLEL', parallel=None) == 0)
    if not_par_config:
        return False

    return True


def USE_GPU():
    if os.environ.get('SSQ_GPU', '0') == '1':
        if torch is None or cupy is None:
            raise ValueError("'SSQ_GPU' requires PyTorch and CuPy installed.")
        return True
    return False


GDEFAULTS = _get_gdefaults()
