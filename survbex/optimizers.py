import json
import logging
import math
import time
from typing import Tuple, Union, Callable, Literal, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sksurv.functions import StepFunction
from sksurv.metrics import concordance_index_censored

logging.basicConfig()

from survbex.estimators import BeranModel as BeranModelNp
from survlimepy.utils.optimisation import OptFuncionMaker
from survlimepy.utils.predict import predict_wrapper
import scipy


def callback(xk) -> bool:
    return False if not all(xk < 1e-10) else True


def scipy_optimization(
        X: np.ndarray,
        neighbours: np.ndarray,
        neighbours_val: np.ndarray,
        bbox_neigh_s: np.ndarray,
        bbox_neigh_val_s: np.ndarray,
        data_point: np.ndarray,
        data_point_S: np.ndarray,
        kernel_name: str,
        kernel_width: float,
        w: np.ndarray,
        w_val: np.ndarray,
        y_event_times: np.ndarray,
        y_events: np.ndarray,
        req_times: np.ndarray,
        criterion,
        lr: float,
        method: str,
        use_tdeltas: bool,
        v_mode: str,
        max_iter: int,
        hessian=None,
        mode='log',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger = logging.getLogger('grad scipy optimizer')
    logger.setLevel(level=logging.DEBUG)

    b = np.ones(neighbours.shape[1])

    y_events = np.array(y_events)
    y_event_times = np.array(y_event_times)

    # neighbours normalization
    # for fi in range(neighbours.shape[1]):
    #     neighbours[:, fi] -= neighbours[:, fi].min()
    #     neighbours[:, fi] /= neighbours[:, fi].max()

    beran_model = BeranModelNp(kernel_width=kernel_width * bbox_neigh_s.std(axis=0).std(), kernel_name=kernel_name)
    beran_model.fit(X=X, b=b, y_event_times=y_event_times, y_events=y_events)
    if use_tdeltas:
        t_deltas = beran_model.event_times_[1:] - beran_model.event_times_[:-1]
    else:
        t_deltas = np.ones_like(beran_model.event_times_[1:])
    t_deltas_train = np.ones((len(neighbours), 1)) @ t_deltas[np.newaxis]
    t_deltas_val = np.ones((len(neighbours_val), 1)) @ t_deltas[np.newaxis]
    train_loss_history = []
    val_loss_history = []
    b_history = []
    s_neigh_history = []
    s_neigh_val_history = []
    s_dp_history = []
    no_conv_ctr = [0]

    if mode in ['log', 'log_seria', 'log_external']:
        if mode == 'log':
            pred_fn = beran_model.predict_log_torch_optimized
        elif mode == 'log_seria':
            pred_fn = beran_model.predict_log_seria_torch_optimized
        elif mode == 'log_external':
            pred_fn = lambda xps: np.log(beran_model.predict_survival_torch_optimized(xps=xps) + epsilon)
        else:
            raise Exception(f"Unexpected mode = {mode}")

        epsilon = 1e-2
        bbox_neigh_train_f = np.log(bbox_neigh_s + epsilon)
        bbox_neigh_val_f = np.log(bbox_neigh_val_s + epsilon)

        bbox_neigh_train_f[bbox_neigh_train_f == 0] = np.abs(bbox_neigh_train_f[bbox_neigh_train_f != 0]).min()
        bbox_neigh_val_f[bbox_neigh_val_f == 0] = np.abs(bbox_neigh_val_f[bbox_neigh_val_f != 0]).min()
        if v_mode == 'no':
            v_train = np.ones((len(neighbours), len(req_times)))
            v_val = np.ones((len(neighbours_val), len(req_times)))
        elif v_mode == 'mult':
            H_neigh = 1 - bbox_neigh_s
            H_neigh_val = 1 - bbox_neigh_val_s
            v_train = H_neigh * np.log(H_neigh + epsilon)
            v_val = H_neigh_val * np.log(H_neigh_val + epsilon)
        elif v_mode == 'div':
            H_neigh = 1 - bbox_neigh_s
            H_neigh_val = 1 - bbox_neigh_val_s
            v_train = H_neigh / np.log(H_neigh + 1 + epsilon)
            v_val = H_neigh_val / np.log(H_neigh_val + 1 + epsilon)
        else:
            raise Exception(f"Undefined v_mode = {v_mode}")
    elif mode in ['surv', 'surv_seria']:
        bbox_neigh_train_f = bbox_neigh_s
        bbox_neigh_val_f = bbox_neigh_val_s
        if mode == 'surv':
            pred_fn = beran_model.predict_survival_torch_optimized
        elif mode == 'surv_seria':
            pred_fn = lambda xps: np.exp(beran_model.predict_log_seria_torch_optimized(xps=xps))
        else:
            raise Exception(f'Undefined mode = {mode}')

        assert v_mode == 'no_surv'
        v_train = np.ones((len(neighbours), len(req_times)))
        v_val = np.ones((len(neighbours_val), len(req_times)))
    else:
        raise Exception(f"Undefined mode = {mode}")

    def target_func(b_new):
        for i in range(len(b)):
            b[i] = b_new[i]

        beran_neigh_train_f = pred_fn(xps=neighbours)
        beran_neigh_val_f = pred_fn(xps=neighbours_val)
        beran_dp_s = pred_fn(xps=data_point)

        train_loss = criterion(y_true=bbox_neigh_train_f, y_pred=beran_neigh_train_f, t_deltas=t_deltas_train,
                               v=v_train, sample_weight=w, b=b) * lr
        val_loss = criterion(y_true=bbox_neigh_val_f, y_pred=beran_neigh_val_f, t_deltas=t_deltas_val, v=v_val,
                             sample_weight=w_val, b=b) * lr
        if math.isnan(train_loss):
            train_loss = val_loss = -1
        elif math.isinf(train_loss):
            train_loss = val_loss = -2

        s_neigh_history.append(beran_neigh_train_f)
        s_neigh_val_history.append(beran_neigh_val_f)
        s_dp_history.append(beran_dp_s)

        ret_val = train_loss
        if len(train_loss_history) > max_iter and train_loss > np.min(train_loss_history):
            no_conv_ctr[0] += 1
            logger.debug(f'no_conv_ctr = {no_conv_ctr[0]}')
            if no_conv_ctr[0] >= 5:
                ret_val = 0
        if np.abs(b).mean() > (1e12 / kernel_width):
            ret_val = 0

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        beran_train_f = pred_fn(xps=X)
        cindex_train = concordance_index_censored(
            event_indicator=y_events, event_time=y_event_times,
            estimate=1 / beran_train_f.mean(axis=-1)
        )[0]
        cindex_target = concordance_index_censored(
            event_indicator=np.ones(len(bbox_neigh_train_f), dtype=np.bool_),
            event_time=bbox_neigh_train_f.mean(axis=-1),
            estimate=1 / beran_neigh_train_f.mean(axis=-1)
        )[0]

        b_history.append(b_new)
        logger.debug(f'{len(train_loss_history)}:weights              = {b}')
        logger.debug(f'{len(train_loss_history)}:weights sum          = {abs(b).sum()}')
        logger.debug(f'{len(train_loss_history)}:train loss           = {train_loss:.4f}')
        logger.debug(f'{len(train_loss_history)}:target cindex        = {cindex_target:.4f}')
        logger.debug(f'{len(train_loss_history)}:train cindex         = {cindex_train:.4f}')
        logger.debug(f'{len(train_loss_history)}:val loss             = {val_loss:.4f}')

        return ret_val

    x0 = np.ones(len(b)) / len(b)

    # l2
    # l2_constraints = scipy.optimize.LinearConstraint(
    #     A=np.identity(len(b)),
    #     lb=np.ones(len(b)) * (-5),
    #     ub=np.ones(len(b)) * 5
    # )

    all_constraints = [
        # l2_constraints
    ]

    # hess = lambda x: np.zeros(len(x))
    hess = None
    res = scipy.optimize.minimize(
        target_func,
        x0=x0,
        constraints=all_constraints,
        method=method,
        hess=hess,
        # tol=1e-6,
        # callback=callback
        options=dict(maxiter=max_iter, disp=True)
    )

    return res.x, \
        np.array(train_loss_history), \
        np.array(val_loss_history), \
        np.array(b_history), \
        np.array(s_neigh_history), \
        np.array(s_neigh_val_history), \
        np.array(s_dp_history)


def mse(y_true, y_pred, t_deltas, sample_weight, v, b):
    y_true = y_true[:, 1:]
    y_pred = y_pred[:, 1:]
    v = v[:, 1:]
    assert y_true.shape == y_pred.shape == t_deltas.shape == v.shape
    return ((np.abs(v) * t_deltas * (y_true - y_pred) ** 2) * sample_weight[:, np.newaxis]).mean()


def mse_l2(y_true, y_pred, t_deltas, sample_weight, v, b, l=0.001):
    return mse(y_true=y_true, y_pred=y_pred, t_deltas=t_deltas, sample_weight=sample_weight, v=v, b=b) \
        + l * math.sqrt(sum(b ** 2))


class GradientOptimizer(OptFuncionMaker):
    def __init__(self, training_features: np.ndarray,
                 training_events: Union[np.ndarray, pd.Series, List[Union[bool, float, int]]],
                 training_times: Union[np.ndarray, pd.Series, List[Union[float, int]]], kernel_width: float,
                 neighbours: np.ndarray, neighbours_transformed: Union[np.ndarray, pd.DataFrame],
                 neighbours_val: np.ndarray, neighbours_val_transformed: Union[np.ndarray, pd.DataFrame],
                 num_samples: int, data_point: np.ndarray, predict_fn: Callable,
                 type_fn: Literal["survival", "cumulative"], functional_norm: Union[float, str],
                 grid_info_file: str, max_iter: int,
                 model_output_times: Optional[np.ndarray] = None,
                 H0: Optional[Union[np.ndarray, List[float], StepFunction]] = None,
                 max_difference_time_allowed: Optional[float] = None, max_hazard_value_allowed: Optional[float] = None,
                 verbose: bool = True):
        super().__init__(training_features, training_events, training_times, kernel_width, neighbours,
                         neighbours_transformed, num_samples, data_point, predict_fn, type_fn, functional_norm,
                         model_output_times, H0, max_difference_time_allowed, max_hazard_value_allowed, verbose)
        self.grid_info_file = grid_info_file
        self.max_iter = max_iter
        self.neighbours_val = neighbours_val
        self.neighbours_val_transformed = neighbours_val_transformed

    def _get_weights(self, neighbours: np.ndarray, data_point: np.ndarray):
        """Compute the weights of each individual.

        Args:
            None

        Returns:
            w (np.ndarray): the weights for each individual.
        """
        # Compute weights.
        distances = pairwise_distances(
            neighbours, data_point, metric=self.weighted_euclidean_distance
        ).ravel()
        weights = self.kernel_fn(distances)
        # w = np.reshape(weights, newshape=(self.num_samples, 1))
        return weights

    def solve_problem(self):
        scaler = MinMaxScaler(feature_range=(1e-5, 1 - 1e-5))
        scaler.fit(np.vstack([self.training_features, self.neighbours, self.neighbours_val, self.data_point]))
        self.training_features = scaler.transform(self.training_features)
        self.neighbours = scaler.transform(self.neighbours)
        self.neighbours_val = scaler.transform(self.neighbours_val)
        self.data_point = scaler.transform(self.data_point)

        assert self.training_features.min() > 0 and self.training_features.max() < 1
        assert self.neighbours.min() > 0 and self.neighbours.max() < 1
        assert self.data_point.min() > 0 and self.data_point.max() < 1

        # Get predictions
        bbox_neigh_s = predict_wrapper(
            predict_fn=self.predict_fn,
            data=self.neighbours_transformed,
            unique_times_to_event=self.unique_times_to_event,
            model_output_times=self.model_output_times,
        )
        self.bbox_neigh_s = bbox_neigh_s

        bbox_neigh_val_s = predict_wrapper(
            predict_fn=self.predict_fn,
            data=self.neighbours_val_transformed,
            unique_times_to_event=self.unique_times_to_event,
            model_output_times=self.model_output_times,
        )
        self.bbox_neigh_val_s = bbox_neigh_val_s

        data_point_S = predict_wrapper(
            predict_fn=self.predict_fn,
            data=self.data_point,
            unique_times_to_event=self.unique_times_to_event,
            model_output_times=self.model_output_times,
        )
        # draw_points_tsne(
        #     pt_groups=[self.training_features, self.neighbours, self.neighbours_val, self.data_point],
        #     names=['train', 'neighbours', 'neighbours val', 'data point'],
        #     colors=[None, None, None, 'red'],
        #     path=f"{RES_DIR}/tsne_approx_points_code_cl={CONFIG['COX_CL_I']}.png"
        # )
        # draw_points_tsne(
        #     pt_groups=[self.training_features, self.data_point],
        #     names=['train', 'data point'],
        #     colors=[None, 'red'],
        #     path=f"{RES_DIR}/tsne_approx_points_code_no_neigh_cl={CONFIG['COX_CL_I']}.png"
        # )
        w_neighs = self._get_weights(neighbours=self.neighbours, data_point=self.data_point)
        w_neighs_val = self._get_weights(neighbours=self.neighbours_val, data_point=self.data_point)
        common_args = dict(
            X=self.training_features,
            neighbours=self.neighbours,
            neighbours_val=self.neighbours_val,
            bbox_neigh_s=bbox_neigh_s,
            bbox_neigh_val_s=bbox_neigh_val_s,
            data_point=self.data_point,
            data_point_S=data_point_S,
            w=w_neighs,
            w_val=w_neighs_val,
            req_times=self.unique_times_to_event,
            y_event_times=self.training_times,
            y_events=self.training_events,
            max_iter=self.max_iter
        )

        ################################################################################################################
        # ---------------------------------------- scipy optimization --------------------------------------------------
        ################################################################################################################
        def mse_l2_wrapper(l):
            res_lamda = lambda **args: mse_l2(**args, l=l)
            res_lamda.__name__ = f'mse_l2={l}'
            return res_lamda

        common_beran_grid = dict(
            criterion=[
                # *[mse_l2_wrapper(l=l) for l in [1e0, 1e-2, 1e-4, 1e-6]],
                mse,
            ],
            kernel_width=[
                # 1e-2, 1e-3, 1e-4
                0.1,
            ],
            kernel_name=[
                # 'gaussian',
                'triangle'
            ],
            method=[
                # 'Nelder-Mead',
                'BFGS',
                # 'CG',
                # 'Powell',
            ],
            use_tdeltas=[
                True,
                # False
            ]
        )
        scipy_grid = ParameterGrid(
            [
                dict(
                    **common_beran_grid,
                    mode=['surv'],
                    lr=[1e0, 1e2, 1e4, 1e6, 1e8, 1e10],
                    v_mode=['no_surv']
                ),
                # dict(
                #     **common_beran_grid,
                #     mode=['log_external', 'log_seria'],
                #     lr=[1e2, 1e3, 1e4, 1e5],
                #     v_mode=['div', 'mult', 'no']
                # )
            ]
        )
        logger = logging.getLogger('grad scipy optimizer')
        logger.setLevel(level=logging.DEBUG)

        grid_list = []
        for grid_i, grid_args in enumerate(scipy_grid):
            logger.debug(f"scipy optimization {grid_i + 1}/{len(scipy_grid)}")
            logger.debug(f'grid_args = {grid_args}')
            start_time = time.time()
            curr_b, \
                train_loss_history, val_loss_history, b_history, \
                s_neigh_history, s_neigh_val_history, s_dp_history = \
                scipy_optimization(**common_args, **grid_args)
            optim_time = time.time() - start_time
            best_train_loss_id = np.argmin(train_loss_history)
            best_val_loss_id = np.argmin(val_loss_history)
            grid_list.append(
                dict(
                    beran_coefs=curr_b,
                    bbox_neigh_s=bbox_neigh_s,
                    bbox_neigh_val_s=bbox_neigh_val_s,
                    bbox_dp_s=data_point_S,
                    beran_train_loss_history=train_loss_history,
                    beran_val_loss_history=val_loss_history,
                    beran_b_history=b_history,
                    beran_s_neigh_history=np.array([
                        s_neigh_history[0],
                        s_neigh_history[best_train_loss_id],
                        s_neigh_history[best_val_loss_id]
                    ]),
                    beran_s_dp_history=np.array([
                        s_dp_history[0],
                        s_dp_history[best_train_loss_id],
                        s_dp_history[best_val_loss_id]
                    ]),
                    grid_args=json.dumps({**grid_args, 'criterion': grid_args['criterion'].__name__}),
                    training_features=self.training_features,
                    training_times=np.array(self.training_times),
                    training_events=np.array(self.training_events),
                    unique_times_to_event=self.unique_times_to_event,
                    neighbours=self.neighbours,
                    neighbours_val=self.neighbours_val,
                    data_point=self.data_point,
                    optim_time=optim_time
                )
            )
        grid_res_df = pd.DataFrame(grid_list)

        ################################################################################################################
        # ---------------------------------- SELECT BEST B AS A BEST APPROXIMATION -------------------------------------
        ################################################################################################################
        best_b_by_conf = [b_history[np.argmin(loss_history)]
                          for loss_history, b_history in grid_res_df[['beran_train_loss_history', 'beran_b_history']]
                          .itertuples(index=False)
                          ]
        grid_argss = [json.loads(grid_args) for grid_args in list(grid_res_df['grid_args'])]

        t_deltas = self.unique_times_to_event[1:] - self.unique_times_to_event[:-1]
        t_deltas_neigh = np.ones((len(self.neighbours), 1)) @ t_deltas[np.newaxis]
        v_neigh = np.ones((len(self.neighbours), len(self.unique_times_to_event)))
        mse_by_grid = []
        for b, grid_args in zip(best_b_by_conf, grid_argss):
            beran_model = BeranModelNp(
                kernel_width=grid_args['kernel_width'] * bbox_neigh_s.std(axis=0).std(),
                kernel_name=grid_args['kernel_name']
            )
            beran_model.fit(
                X=self.training_features,
                b=b,
                y_event_times=np.array(self.training_times),
                y_events=np.array(self.training_events)
            )
            beran_pred = beran_model.predict_survival_torch_optimized(xps=self.neighbours)
            mse_by_grid.append(
                mse(y_true=bbox_neigh_s, y_pred=beran_pred, t_deltas=t_deltas_neigh, sample_weight=w_neighs,
                    v=v_neigh, b=b)
            )
        best_b = best_b_by_conf[np.argmin(mse_by_grid)]

        ################################################################################################################
        # ---------------------------------- SAVE DETAILED GRID DF FOR POSSIBLE ANALYSIS -------------------------------
        ################################################################################################################
        np_keys = [
            'beran_coefs',
            'bbox_neigh_s', 'bbox_neigh_val_s', 'bbox_dp_s',
            'beran_train_loss_history', 'beran_val_loss_history',
            'beran_b_history',
            'beran_s_neigh_history', 'beran_s_dp_history',
            'training_features', 'training_times', 'training_events',
            'unique_times_to_event',
            'neighbours', 'neighbours_val', 'data_point'
        ]
        for key in np_keys:
            grid_res_df[key] = [json.dumps(arr_like.tolist()) for arr_like in grid_res_df[key]]

        grid_res_df.to_csv(self.grid_info_file)

        return best_b
