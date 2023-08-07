import json
import re
from pathlib import Path

import numpy as np
import os

with open('config_synth_data.json') as fp:
    CONFIG = json.load(fp)
    CONFIG['COX_COEFS_CLS'] = np.array(CONFIG['COX_COEFS_CLS'])
    assert len(CONFIG['COX_COEFS_CLS'].shape) == 2, 'wrong dimension of importances matrix'

# CONFIG_STR = ','.join([f'{key}={value}' for key, value in CONFIG.items()])
CONFIG_STR = ','.join([f'{key}={value}' for key, value in CONFIG.items() if key != 'COX_COEFS_CLS'])
BERAN_EXP_PREF = Path(
    f'all_exp_comparison={CONFIG["COX_COEFS_CLS"].shape[0]}_kernels_COX_COEFS=[{CONFIG["COX_COEFS_CLS"].shape[1]} features],' + CONFIG_STR)

RES_DIR = Path('beran_res').joinpath(BERAN_EXP_PREF)
RES_DIR.mkdir(exist_ok=True, parents=True)


def get_v() -> int:
    subdirs = [x[0] for x in os.walk(RES_DIR) if bool(re.match(".+v[0-9]+$", x[0]))]
    if len(subdirs) == 0:
        return 0
    else:
        int_from_str_path = lambda x: int(Path(x).name.split('v')[1])
        return int_from_str_path(sorted(subdirs, key=int_from_str_path)[-1]) + 1


RES_DIR = RES_DIR.joinpath(f'v{get_v()}')

RES_DIR.mkdir(exist_ok=True, parents=True)


def serialize_object(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


with open(f"{RES_DIR}/config.json", 'w') as fp:
    json.dump(fp=fp, obj={key: serialize_object(value) for key, value in CONFIG.items()})
