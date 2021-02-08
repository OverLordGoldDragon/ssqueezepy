# -*- coding: utf-8 -*-
import os
import inspect
import logging

logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
NOTE = lambda msg: logging.warning("NOTE: %s" % msg)  # else it's mostly ignored


def float_if_number(s):
    """If float works, so should int."""
    try:
        return float(s)
    except ValueError:
        return s


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
        GDEFAULTS[module][obj][key] = float_if_number(value)


def gdefaults(module_and_obj=None, **kw):
    if module_and_obj is None:
        obj = inspect.stack()[1][3]
        module = inspect.stack()[1][1].split(os.path.sep)[-1].rstrip('.py')
    else:
        module, obj = module_and_obj.split('.')
    if not isinstance(kw, dict):
        raise TypeError("`kw` must be dict (got %s)" % type(kw))

    for key, value in kw.items():
        if value is None:
            if module not in GDEFAULTS:
                WARN(f"module {module} not found in GDEFAULTS (see configs.ini)")
            elif obj not in GDEFAULTS:
                WARN(f"object {obj} not found in GDEFAULTS (see configs.ini)")
            else:
                kw[key] = GDEFAULTS[module][obj].get(key, value)
    return kw


def handle_defaults(fn):
    fn_name = inspect.stack()[1][4][0].lstrip('def ').split('(')[0]

    def wrap(*args, **kwargs):
        sig = str(inspect.signature(fn)).strip('()').split(', ')
        kw = {}
        for arg in sig:
            if '=' in arg:
                name, value = arg.split('=')
                kw[name] = value if value != 'None' else None

        module = inspect.stack()[1][1].split(os.path.sep)[-1].rstrip('.py')
        # obj = inspect.stack()[1][3]
        obj = fn_name
        module_and_obj = module + '.' + obj
        defaults = gdefaults(module_and_obj, **kw)

        kwargs = kwargs.copy()
        for k, v in defaults.items():
            if kwargs.get(k, None) is None:
                kwargs[k] = v
        return fn(*args, **kwargs)
    return wrap
