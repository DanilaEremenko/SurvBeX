import json
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

from pandas.core.dtypes.common import is_categorical_dtype, is_string_dtype
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sksurv.datasets import load_gbsg2, load_veterans_lung_cancer, load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis

from core.drawing import draw_points_tsne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from core.cox_generator import CoxGenerator
from survbex.explainers import SurvBexExplainer
from sksurv.util import Surv


def explain_with_logging(explainer, model, config, ex_data: pd.Series, test_size: int, pt_i: int, repeat_i: int,
                         mode: str):
    print(f'{mode}: pt = {pt_i}/{test_size}, repeat = {repeat_i}')
    optimizer = 'convex' if mode == 'cox' else 'gradient'
    return explainer.explain_instance(
        data_row=ex_data,
        predict_fn=model.predict_survival_function,
        num_samples=config['NEIGH_SIZE'],
        num_val_samples=config['NEIGH_VAL_SIZE'],
        type_fn='survival',
        optimizer=optimizer
    )


def save_train_test_data(x_train, y_train, x_test, y_test):
    x_train.to_csv(f'{RES_DIR}/train_x.csv', index=False)
    y_train_df = pd.DataFrame({
        'event': [pair[0] for pair in y_train],
        'event_time': [pair[1] for pair in y_train]}
    )
    y_train_df.to_csv(f'{RES_DIR}/train_y.csv', index=False)

    x_test.to_csv(f'{RES_DIR}/test_x.csv', index=False)
    y_test_df = pd.DataFrame({
        'event': [pair[0] for pair in y_test],
        'event_time': [pair[1] for pair in y_test]}
    )
    y_test_df.to_csv(f'{RES_DIR}/test_y.csv', index=False)


def prepare_real_ds_by_name(name: str):
    if name == 'veterans':
        x_data, y_data = load_veterans_lung_cancer()
    elif name == 'gbsg2':
        x_data, y_data = load_gbsg2()
    elif name == 'whas500':
        x_data, y_data = load_whas500()
    else:
        raise Exception(f"Undefined ds = {name}")

    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in x_data.keys()
        if is_string_dtype(x_data[key]) or is_categorical_dtype(x_data[key])
    }

    for key, le in le_dict.items():
        x_data[key] = le.fit_transform(x_data[key])

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

    save_train_test_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    return [x_train, y_train], [x_test, y_test]


def get_cox_data(coefs: np.ndarray, cl_i: int):
    cox_generator = CoxGenerator(coefs=coefs)
    x_cox_train, x_cox_test, y_cox_train, y_cox_test = train_test_split(
        *cox_generator.generate_data(size=CONFIG['TRAIN_SIZE'], censored_part=0.2),
        train_size=0.9
    )

    x_cox_train = pd.DataFrame(x_cox_train, columns=[f'f{i + 1}' for i in range(len(coefs))])
    x_cox_train['cl_i'] = cl_i
    x_cox_test = pd.DataFrame(x_cox_test, columns=[f'f{i + 1}' for i in range(len(coefs))])
    x_cox_test['cl_i'] = cl_i

    return [x_cox_train, y_cox_train], [x_cox_test, y_cox_test]


def generate_cox_ds():
    cox_clusters = [get_cox_data(coefs=cox_coefs, cl_i=cl_i) for cl_i, cox_coefs in
                    enumerate(CONFIG['COX_COEFS_CLS'])]

    cox_clusters = [
        (
            [cox_cluster[0][0] + 2.0 / len(cox_clusters) * cl_i, cox_cluster[0][1]],
            [cox_cluster[1][0] + 2.0 / len(cox_clusters) * cl_i, cox_cluster[1][1]]
            # [cox_cluster[0][0] + 1. * cl_i, cox_cluster[0][1]],
            # [cox_cluster[1][0] + 1. * cl_i, cox_cluster[1][1]]
        )
        for cl_i, cox_cluster in enumerate(cox_clusters)
    ]

    all_train = [
        pd.concat([cox_cluster[0][0] for cox_cluster in cox_clusters]),
        np.hstack([cox_cluster[0][1] for cox_cluster in cox_clusters])
    ]

    all_test = [
        pd.concat([cox_cluster[1][0] for cox_cluster in cox_clusters]),
        np.hstack([cox_cluster[1][1] for cox_cluster in cox_clusters])
    ]

    save_train_test_data(x_train=all_train[0], y_train=all_train[1], x_test=all_test[0], y_test=all_test[1])

    with open(f"{RES_DIR}/config.json", 'w') as fp:
        json.dump(obj={**CONFIG, 'COX_COEFS_CLS': CONFIG['COX_COEFS_CLS'].tolist()}, fp=fp)

    draw_points_tsne(
        pt_groups=[cox_cluster[0][0].drop(columns=['cl_i']) for cox_cluster in cox_clusters],
        names=['cl1', 'cl2', 'cl3'],
        colors=[None] * len(cox_clusters),
        path=f'{RES_DIR}/cl_pts.png'
    )

    return all_train, all_test


def drop_cl_info(df):
    if CONFIG['DS'] == 'cox':
        return df.drop(columns=['cl_i'])
    else:
        return df


def load_ds():
    all_train = [
        pd.read_csv(f'{RES_DIR}/train_x.csv'),
        Surv.from_dataframe(data=pd.read_csv(f'{RES_DIR}/train_y.csv'), event='event', time='event_time')
    ]

    all_test = [
        pd.read_csv(f'{RES_DIR}/test_x.csv'),
        Surv.from_dataframe(data=pd.read_csv(f'{RES_DIR}/test_y.csv'), event='event', time='event_time')
    ]

    return all_train, all_test


if __name__ == '__main__':

    with open('config_real_data.json') as fp:
        CONFIG = json.load(fp=fp)
        CONFIG['COX_COEFS_CLS'] = np.array(CONFIG['COX_COEFS_CLS'])

    CONFIG_STR = ','.join([f'{key}={value}' for key, value in CONFIG.items() if key != 'COX_COEFS_CLS'])
    RES_DIR = Path('beran_res_real').joinpath(CONFIG_STR)
    RES_DIR.mkdir(exist_ok=True, parents=True)

    if CONFIG['DS'] in ('veterans', 'gbsg2', 'whas500', 'cox'):
        if RES_DIR.joinpath('train_x.csv').exists():
            all_train, all_test = load_ds()
        else:
            all_train, all_test = prepare_real_ds_by_name(name=CONFIG['DS'])
    else:
        raise Exception(f'Undefined ds = {CONFIG["DS"]}')

    draw_points_tsne(
        pt_groups=[drop_cl_info(all_train[0]), drop_cl_info(all_test[0])],
        names=['train_points', 'test_points'],
        colors=[None] * 2,
        path=f'{RES_DIR}/train_test_pts.png'
    )

    if CONFIG['DS'] == 'cox':
        draw_points_tsne(
            pt_groups=[all_test[0][all_test[0]['cl_i'] == cl_i] for cl_i in all_test[0]['cl_i'].unique()],
            names=[f"cl={i}" for i, _ in enumerate(all_test[0]['cl_i'].unique())],
            colors=[None] * len(all_test[0]['cl_i'].unique()),
            path=f'{RES_DIR}/clusters_test_pts.png'
        )

    scaler = MinMaxScaler(feature_range=(1e-5, 1 - 1e-5))
    scaler.fit(pd.concat([all_train[0], all_test[0]]))
    feature_keys = all_test[0].keys()
    all_train[0] = pd.DataFrame(scaler.transform(all_train[0]), columns=feature_keys)
    all_test[0] = pd.DataFrame(scaler.transform(all_test[0]), columns=feature_keys)

    # model = CoxPHSurvivalAnalysis()
    if CONFIG['BBOX'] == 'rf':
        model = RandomSurvivalForest(n_estimators=100, max_samples=min(500, len(all_train[0])), max_depth=8)
        model.fit(drop_cl_info(all_train[0]), all_train[1])
    elif CONFIG['BBOX'] == 'cox':
        model = CoxPHSurvivalAnalysis(alpha=1)
        # model = CoxWrapperLifeLines()
        model.fit(drop_cl_info(all_train[0]), all_train[1])
    else:
        raise Exception(f"Undefined bbox = {CONFIG['BBOX']}")

    # Use SurvLimeExplainer class to find the feature importance
    training_features = all_train[0]
    training_events = [event for event, _ in all_train[1]]
    training_times = [time for _, time in all_train[1]]

    test_events = [event for event, _ in all_test[1]]
    test_times = [time for _, time in all_test[1]]

    cindex_train = model.score(drop_cl_info(all_train[0]), all_train[1])
    print(f'cindex train = {cindex_train}')
    cindex_test = model.score(drop_cl_info(all_test[0]), all_test[1])
    print(f'cindex test = {cindex_test}')

    explainer = SurvBexExplainer(
        training_features=drop_cl_info(training_features),
        training_events=training_events,
        training_times=training_times,
        model_output_times=model.event_times_,
        kernel_width=CONFIG['KERNEL_WIDTH']
    )

    # explanation variable will have the computed SurvLIME values
    # (test_size, num_repeats, feature_size)

    ####################################################################################################################
    # ------------------------------------------------ SurvLIME --------------------------------------------------------
    ####################################################################################################################
    cox_explanations = np.array(
        [
            [
                explain_with_logging(
                    explainer=explainer,
                    model=model,
                    config=CONFIG,
                    ex_data=test_pt,
                    test_size=len(all_test[0]),
                    pt_i=test_pt_i,
                    repeat_i=repeat_i,
                    mode='cox'
                )
                for repeat_i in range(3)
            ]
            for test_pt_i, test_pt in drop_cl_info(all_test[0]).iterrows()
        ]
    )
    np.save(f'{RES_DIR}/cox_explanations.npy', cox_explanations)

    # cox_explanations = np.load(f'{RES_DIR}/cox_explanations.npy')

    ####################################################################################################################
    # ------------------------------------------------ SurvBeX ---------------------------------------------------------
    ####################################################################################################################
    beran_args = [

        (
            explainer,
            model,
            CONFIG,
            test_pt,
            len(all_test[0]),
            test_pt_i,
            repeat_i,
            'beran'
        )
        for test_pt_i, test_pt in drop_cl_info(all_test[0]).iterrows()
        for repeat_i in range(3)
    ]

    with Pool(processes=15) as pool:
        beran_explanations = pool.starmap(explain_with_logging, beran_args)

    np.save(f'{RES_DIR}/beran_explanations.npy', np.array(beran_explanations))

    # beran_explanations = np.load(f'{RES_DIR}/beran_explanations.npy')
