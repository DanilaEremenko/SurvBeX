import json
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import nelson_aalen_estimator, kaplan_meier_estimator

from core.cox_wrapper import CoxFairBaseline
from core.drawing import draw_points_tsne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from exp_config import CONFIG, RES_DIR
from core.cox_generator import CoxGenerator
from survshap import SurvivalModelExplainer, ModelSurvSHAP
from survbex.estimators import BeranModel
from survbex.explainers import SurvBexExplainer


########################################################################################################################
# ------------------------------------------------ PREPARE DATA --------------------------------------------------------
########################################################################################################################
def get_cox_data(coefs: np.ndarray):
    cox_generator = CoxGenerator(coefs=coefs)
    x_cox_train, x_cox_test, y_cox_train, y_cox_test = train_test_split(
        *cox_generator.generate_data(size=CONFIG['TRAIN_SIZE'], censored_part=0.2),
        train_size=0.7
    )

    x_cox_train = pd.DataFrame(x_cox_train, columns=[f'f{i + 1}' for i in range(len(coefs))])
    x_cox_test = pd.DataFrame(x_cox_test, columns=[f'f{i + 1}' for i in range(len(coefs))])

    return [x_cox_train, y_cox_train], [x_cox_test, y_cox_test]


# np.random.seed(42)
# train, test = get_veterans_data()
cox_clusters = [get_cox_data(coefs=cox_coefs) for cox_coefs in CONFIG['COX_COEFS_CLS']]

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

# Use SurvLimeExplainer class to find the feature importance
training_events = np.array([event for event, _ in all_train[1]])
training_times = np.array([time for _, time in all_train[1]])
training_features = all_train[0]

test_events = np.array([event for event, _ in all_test[1]])
test_times = np.array([time for _, time in all_test[1]])
test_features = all_test[0]

with open(f'{RES_DIR}/dataset.json', 'w+') as fp:
    json.dump(fp=fp, obj=dict(
        training_features=training_features.to_dict(orient='raw'),
        training_events=training_events.tolist(),
        training_times=training_times.tolist(),
        test_features=test_features.to_dict(orient='raw'),
        test_events=test_events.tolist(),
        test_times=test_times.tolist()
    ))

########################################################################################################################
# ------------------------------------------------ BUILD BBOX ----------------------------------------------------------
########################################################################################################################
if CONFIG['BBOX'] == 'rf':
    model = RandomSurvivalForest(n_estimators=100, max_samples=min(500, len(all_train[0])), max_depth=8)
    model.fit(all_train[0], all_train[1])
    pred_surv_fn = model.predict_survival_function
    pred_hazard_fn = model.predict_cumulative_hazard_function
    pred_risk_fn = model.predict
elif CONFIG['BBOX'] == 'beran':
    assert len(CONFIG['COX_COEFS_CLS']) == 1
    model = BeranModel(kernel_width=250, kernel_name='gaussian')
    model.fit(X=all_train[0].to_numpy(), b=CONFIG['COX_COEFS_CLS'][0],
              y_events=training_events, y_event_times=training_times)


    def surv_np_to_step_surv(surv_arr: np.ndarray):
        return np.array([StepFunction(x=model.unique_times_, y=sample) for sample in surv_arr])


    pred_surv_fn = lambda X: surv_np_to_step_surv(model.predict_survival_torch_optimized(X))
    pred_hazard_fn = lambda X: -np.log(model.predict_survival_torch_optimized(X))
    pred_risk_fn = lambda X: np.sum(pred_hazard_fn(X), axis=1)

elif 'cox' in CONFIG['BBOX']:
    model = CoxPHSurvivalAnalysis(alpha=1)

    model.fit(all_train[0], all_train[1])
    pred_surv_fn = model.predict_survival_function
    pred_hazard_fn = model.predict_cumulative_hazard_function
    pred_risk_fn = model.predict

    if CONFIG['BBOX'] in ['cox_na', 'cox_km']:
        if CONFIG['BBOX'] == 'cox_na':
            cox_fair_baseline = CoxFairBaseline(
                training_events=training_events,
                training_times=training_times,
                baseline_estimator_f=nelson_aalen_estimator
            )
        elif CONFIG['BBOX'] == 'cox_km':
            cox_fair_baseline = CoxFairBaseline(
                training_events=training_events,
                training_times=training_times,
                baseline_estimator_f=kaplan_meier_estimator
            )
        else:
            raise Exception(f'Undefined cox model = {CONFIG["BBOX"]}')

        model.coef_ /= np.abs(model.coef_).sum()
        pred_surv_fn = lambda X: cox_fair_baseline.predict_survival_function(X, cox_coefs=model.coef_)
        pred_hazard_fn = lambda X: cox_fair_baseline.predict_cum_hazard_from_surv_np(X, cox_coefs=model.coef_)
        pred_risk_fn = lambda X: np.dot(X, model.coef_)
    elif CONFIG['BBOX'] != 'cox':
        raise Exception(f'Undefined cox model = {CONFIG["BBOX"]}')
else:
    raise Exception(f"Undefined bbox = {CONFIG['BBOX']}")

cindex_train = concordance_index_censored(
    event_indicator=training_events, event_time=training_times, estimate=pred_risk_fn(training_features))[0]
print(f'cindex train = {cindex_train}')
cindex_test = concordance_index_censored(
    event_indicator=test_events, event_time=test_times, estimate=pred_risk_fn(test_features))[0]
print(f'cindex test = {cindex_test}')

########################################################################################################################
# ------------------------------------------------ SELECT POINTS TO EXPLAIN --------------------------------------------
########################################################################################################################
# draw_comparison(ex_i=random.randint(0, len(test)))
cluster_centroids = [
    cox_cluster[0][0].mean() + all_test[0].std() * CONFIG['DATA_POINT_DEV']
    for cox_cluster in cox_clusters
]

cl_distances = [
    [sum((cl_centroid - fs) ** 2) for fs in all_test[0].to_numpy()]
    for cl_centroid in cluster_centroids
]

exp_test_ids = [np.argmin(distances) for distances in cl_distances]

draw_points_tsne(
    pt_groups=[
        *[cox_cluster[0][0].to_numpy() for cox_cluster in cox_clusters],
        *list(all_test[0].to_numpy()[exp_test_ids])
    ],
    names=[
        *[f'cl{i}' for i, _ in enumerate(cox_clusters)],
        *[f'ex for cl {i}' for i, _ in enumerate(exp_test_ids)]
    ],
    colors=[None] * len(cox_clusters) * 2,
    path=f'{RES_DIR}/clusters.png'
    # path=f'clusters.png'
)

with open(RES_DIR.joinpath("y_true.json"), 'w+') as fp:
    json.dump(
        fp=fp,
        obj=[
            dict(event=bool(all_test[1][ex_i][0]), event_time=all_test[1][ex_i][1])
            for ex_i in exp_test_ids
        ]
    )

########################################################################################################################
# ------------------------------------------------ SurvSHAP ------------------------------------------------------------
########################################################################################################################
surv_shap = SurvivalModelExplainer(model, all_test[0].iloc[exp_test_ids], all_test[1][exp_test_ids],
                                   predict_survival_function=lambda model, X: pred_surv_fn(X))

exp_survshap = ModelSurvSHAP(random_state=42)
exp_survshap.fit(surv_shap)
shap_explanations = np.array(
    [
        [
            imp[1]
            for imp in pt_exp.simplified_result.values
        ]
        for pt_exp in exp_survshap.individual_explanations

    ]
)

with open(RES_DIR.joinpath("explanation_shap.json"), 'w+') as fp:
    json.dump(fp=fp, obj=shap_explanations.tolist())

########################################################################################################################
# ------------------------------------------------ SurvLIME ------------------------------------------------------------
########################################################################################################################
explainer = SurvBexExplainer(
    training_features=training_features,
    training_events=list(training_events),
    training_times=list(training_times),
    model_output_times=model.event_times_,
    kernel_width=CONFIG['KERNEL_WIDTH']
)

cox_explanations = np.array(
    [
        explainer.explain_instance(
            data_row=all_test[0].iloc[ex_i],
            predict_fn=pred_surv_fn,
            num_samples=CONFIG['NEIGH_SIZE'],
            type_fn='survival',
            optimizer='convex'
        )
        for ex_i in exp_test_ids
    ]
)

with open(RES_DIR.joinpath("explanation_cox.json"), 'w+') as fp:
    json.dump(fp=fp, obj=cox_explanations.tolist())

########################################################################################################################
# ------------------------------------------------ SurvBeX -------------------------------------------------------------
########################################################################################################################
beran_explanations = []
for cl_i, ex_i in enumerate(exp_test_ids):
    beran_explanations.append(
        explainer.explain_instance(
            data_row=all_test[0].iloc[ex_i],
            predict_fn=pred_surv_fn,
            num_samples=CONFIG['NEIGH_SIZE'],
            num_val_samples=CONFIG['NEIGH_VAL_SIZE'],
            type_fn='survival',
            optimizer='gradient',
            grid_info_file=f"{RES_DIR}/optimization_cl={cl_i}.csv",
            max_iter=CONFIG['MAX_ITER']

        )
    )

with open(RES_DIR.joinpath("explanation_beran.json"), 'w+') as fp:
    json.dump(
        fp=fp,
        obj=np.array(beran_explanations).tolist()
    )
