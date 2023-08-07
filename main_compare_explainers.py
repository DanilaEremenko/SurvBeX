import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error


class ComparedModel:
    def __init__(self, model_name: str, file_name: str, color: str):
        self.model_name = model_name
        self.file_name = file_name
        self.color = color


compare_models = [
    ComparedModel(model_name='SurvLIME', file_name='explanation_cox.json', color='#ff3333'),
    ComparedModel(model_name='SurvBeX', file_name='explanation_beran.json', color='#90cded'),
    ComparedModel(model_name='SurvSHAP', file_name='explanation_shap.json', color='#FA8128'),
    # ComparedModel(model_name='SurvLIME-RISK', file_name='explanation_risk_cox.json'),
    # ComparedModel(model_name='LIME', file_name='explanation_reg_lime.json')
]


def load_json(path: Path):
    with open(path) as fp:
        return json.load(fp)


b_metrics = {
    'mse': mean_squared_error,
    'kl_div': lambda y_true, y_pred: sum(scipy.special.rel_entr(y_true, y_pred)),
    'cindex': lambda y_true, y_pred: concordance_index(y_true, y_pred),
    'r': lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0, 1]
}


def normalize_coefs(b):
    return abs(b) / sum(abs(b))


def get_metrics_for_model(config_dict: dict, compared_model: ComparedModel):
    model_file = config_dict['config_file'].parent.joinpath(compared_model.file_name)
    if not model_file.exists():
        return []

    with open(model_file) as fp:
        pred_list = json.load(fp)
        return [
            {
                'cluster_i': pt_i,
                # 'pt_pred': pt_pred,
                'model_name': compared_model.model_name,
                **{
                    mname: mfunc(y_true=normalize_coefs(np.array(config_dict['COX_COEFS_CLS'][pt_i])),
                                 y_pred=normalize_coefs(np.array(pt_pred)))
                    for mname, mfunc in b_metrics.items()
                }
            }
            for pt_i, pt_pred in enumerate(pred_list)

        ]


EXP_DIRS = [
    'beran_res/all_exp_comparison=1_kernels_COX_COEFS=[5 features],TRAIN_SIZE=200,NEIGH_SIZE=100,NEIGH_VAL_SIZE=20,BERAN_K_NORM=NO,MAX_ITER=100,DATA_POINT_DEV=0.0,KERNEL_WIDTH=0.4,BBOX=beran',
    'beran_res/all_exp_comparison=1_kernels_COX_COEFS=[5 features],TRAIN_SIZE=200,NEIGH_SIZE=100,NEIGH_VAL_SIZE=20,BERAN_K_NORM=NO,MAX_ITER=100,DATA_POINT_DEV=0.0,KERNEL_WIDTH=0.4,BBOX=cox',
    'beran_res/all_exp_comparison=1_kernels_COX_COEFS=[5 features],TRAIN_SIZE=200,NEIGH_SIZE=100,NEIGH_VAL_SIZE=20,BERAN_K_NORM=NO,MAX_ITER=100,DATA_POINT_DEV=0.0,KERNEL_WIDTH=0.4,BBOX=cox_na',
    'beran_res/all_exp_comparison=1_kernels_COX_COEFS=[5 features],TRAIN_SIZE=200,NEIGH_SIZE=100,NEIGH_VAL_SIZE=20,BERAN_K_NORM=NO,MAX_ITER=100,DATA_POINT_DEV=0.0,KERNEL_WIDTH=0.4,BBOX=rf'
]

if __name__ == '__main__':
    for exp_dir in EXP_DIRS:
        assert Path(exp_dir).exists(), f'{exp_dir} not exists'
        config_files = [
            Path(f"{dir}/{file}")
            for dir, _, files in os.walk(exp_dir)
            for file in files
            if 'config.json' in file
        ]
        config_dicts = [{'config_file': config_file, **load_json(config_file)} for config_file in config_files]

        res_df = pd.DataFrame(
            [
                res_dict
                for config_dict in config_dicts
                for compared_model in compare_models
                for res_dict in get_metrics_for_model(config_dict=config_dict, compared_model=compared_model)
            ]
        )

        # group_df = res_df.groupby(['model_name', 'cluster_i']).mean()
        # group_std_df = res_df.groupby(['model_name', 'cluster_i']).std()
        group_df = res_df.groupby(['model_name']).mean().drop(columns=['cluster_i'])
        group_std_df = res_df.groupby(['model_name']).std().drop(columns=['cluster_i'])

        print(f"------------------------------------------------------------------")
        print(f"\nBBOX {exp_dir.split('BBOX')[-1]}")
        print(group_df.round(3))
        print(group_std_df.round(3))

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        #
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4 * 3, 4))
        # pal = {compared_model.model_name: compared_model.color for compared_model in compare_models}
        # for ax, m_name in zip(axes, b_metrics.keys()):
        #     sns.boxplot(data=res_df, x='model_name', y=m_name, whis=1000, palette=pal,
        #                 order=['SurvLIME', 'SurvSHAP', 'SurvBeX'])
        #     plt.ylabel(f'{m_name}')
        #     plt.xticks(rotation=45)
        #     if m_name in ['cindex']:
        #         plt.ylim((0, 1))
        #     plt.tight_layout()
        #     plt.sca(ax)
        # plt.show()
