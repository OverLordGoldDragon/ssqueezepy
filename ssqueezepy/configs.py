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


path = os.path.join(os.path.dirname(__file__), 'configs.ini')
with open(path, 'r') as f:
    txt = f.read().split('\n')
    txt = txt[:txt.index('#### END')]
    txt = [line.strip(' ') for line in txt if line != '']

# global defaults
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


def gdefaults(module_and_obj=None, get_all=False, as_dict=False, **kw):
    if module_and_obj is None:
        stack = inspect.stack(0)  # `(0)` faster than `()`
        obj = stack[1][3]
        module = stack[1][1].split(os.path.sep)[-1].rstrip('.py')
    else:
        module, obj = module_and_obj.split('.')

    for key, value in kw.items():
        if value is None:
            if module not in GDEFAULTS:
                WARN(f"module {module} not found in GDEFAULTS (see configs.ini)")
            elif obj not in GDEFAULTS[module]:
                WARN(f"object {obj} not found in GDEFAULTS['{module}'] "
                     "(see configs.ini)")
            else:
                kw[key] = GDEFAULTS[module][obj].get(key, value)

    if module in GDEFAULTS and obj in GDEFAULTS[module]:
        # preserve defaults' order
        defaults = {}
        for k, v in GDEFAULTS[module][obj].items():
            if k in kw:
                defaults[k] = kw[k]
            elif get_all:
                defaults[k] = v
        kw = defaults

    if as_dict:
        return kw
    return (kw.values() if len(kw) != 1 else
            list(kw.values())[0])
