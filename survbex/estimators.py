import logging
import math

import numpy as np
import pandas as pd
import torch

logging.basicConfig()


class BeranModel:
    def kernel_gaussian(self, xi: np.ndarray, xj: torch.Tensor, b: torch.Tensor):
        u = xi - xj
        return torch.exp(-torch.mean(self.kernel_width * b * u ** 2))

    def kernel_fake_triangle(self, xi: np.ndarray, xj: torch.Tensor, b: torch.Tensor):
        u = xi - xj
        return torch.exp(-torch.mean(self.kernel_width * b * np.abs(u)))

    def get_kernel_fn_by_name(self, kernel_name: str):
        if kernel_name == 'gaussian':
            return self.kernel_gaussian
        elif kernel_name == 'triangle':
            return self.kernel_fake_triangle
        else:
            raise Exception(f'Unexpected kernel = {kernel_name}')

    def get_torch_kernel_dist_fn_by_name(self, kernel_name: str):
        if kernel_name == 'gaussian':
            return torch.square
        elif kernel_name == 'triangle':
            return torch.abs
        else:
            raise Exception(f'Unexpected kernel = {kernel_name}')

    def __init__(self, kernel_width: float, kernel_name: str, log_epsilon=0.001, verbose=False):
        self.kernel_width = kernel_width
        self.kernel_fn = self.get_kernel_fn_by_name(kernel_name=kernel_name)
        self.kernel_dist_fn = self.get_torch_kernel_dist_fn_by_name(kernel_name=kernel_name)
        self.log_epsilon = log_epsilon
        self.logger = logging.getLogger('beran')
        self.logger.setLevel(level=logging.DEBUG if verbose else logging.ERROR)

    def fit(self, X: np.ndarray, b: np.ndarray, y_event_times: np.ndarray, y_events: np.ndarray):
        # data should be normalized
        for fi in range(X.shape[1]):
            assert X[:, fi].min() >= 0
            assert X[:, fi].max() <= 1

        self.b = b
        sort_idx = np.argsort(y_event_times)
        self.X_sorted = X[sort_idx]
        self.y_ets_sorted = y_event_times[sort_idx]
        self.y_events_sorted = y_events[sort_idx]

        self.X_sorted_torch = torch.tensor(self.X_sorted)
        self.y_ets_sorted_torch = torch.tensor(y_event_times[sort_idx])
        self.y_events_sorted_torch = torch.tensor(y_events[sort_idx])

        self.event_times_ = np.unique(self.y_ets_sorted)
        self.unique_times_ = self.event_times_

        _, idx_start, count = np.unique(self.y_ets_sorted, return_counts=True, return_index=True)
        self.ids_with_repeats = idx_start + (count - 1)

    ####################################################################################################################
    # ------------------------------------------------ PYTHON IMPLEMENTATIONS -------------------------------------------
    ####################################################################################################################
    def K(self, xi: np.ndarray, xj: np.ndarray) -> float:
        norm_b = np.abs(self.b).sum()
        # norm_b = np.linalg.norm(b)
        b_normalized = np.abs(self.b) / norm_b if norm_b != 0 else self.b
        return math.exp(-np.sum(b_normalized * (xi - xj) ** 2) / (2 * self.kernel_width ** 2))

    def K_der1(self, xi: np.ndarray, xj: np.ndarray):
        norm_b = sum(abs(self.b))
        b_normalized = abs(self.b) / norm_b if norm_b != 0 else self.b
        return sum((xi - xj) ** 2) * math.exp(sum(b_normalized * (xi - xj) ** 2))

    def W(self, xp: np.ndarray, xj: np.ndarray, norm: np.ndarray) -> np.ndarray:
        return self.K(xi=xp, xj=xj) / norm

    def Stx(self, xp: np.ndarray) -> np.ndarray:
        # data should be normalized
        for fi in range(self.X_sorted.shape[1]):
            assert 0 <= xp[fi] <= 1, f"xp[{fi}] = {xp[fi]}"

        st = np.ones(1)
        st_horizon = np.zeros(len(self.event_times_))

        norm = sum(np.array([self.K(xi=xp, xj=xk) for xk in self.X_sorted]))
        curr_req_i = 0

        assert len(self.X_sorted) == len(self.y_ets_sorted)
        for i, (xi, y_et, y_event) in enumerate(zip(self.X_sorted, self.y_ets_sorted, self.y_events_sorted)):
            Xjs = self.X_sorted[:max(i - 1, 0)]
            if i % 100 == 0:
                self.logger.debug(f'x{i}/x{len(self.X_sorted)}, S(t) = {st.item():.3f}')
            num = self.W(xp=xp, xj=xi, norm=norm)
            Xjs_w_sum = sum(np.array([self.W(xp=xp, xj=xj, norm=norm) for xj in Xjs]))
            if Xjs_w_sum == 1:
                continue
            denum = 1 - Xjs_w_sum
            st *= (1 - num / denum) if y_event else 1

            if y_et == self.event_times_[curr_req_i]:
                self.logger.debug(f'S({self.y_ets_sorted[curr_req_i]}) = {st.item():.3f} by x{len(Xjs)}')
                st_horizon[curr_req_i] = st
                curr_req_i += 1
                if curr_req_i == len(self.event_times_):
                    return st_horizon

        return st_horizon

    def StxLog(self, xp: np.ndarray) -> np.ndarray:
        # data should be normalized
        for fi in range(self.X_sorted.shape[1]):
            assert 0 <= xp[fi] <= 1, f"xp[{fi}] = {xp[fi]}"

        st = np.zeros(1)
        st_horizon = np.zeros(len(self.event_times_))

        norm = sum(np.array([self.K(xi=xp, xj=xk) for xk in self.X_sorted]))
        curr_req_i = 0

        assert len(self.X_sorted) == len(self.y_ets_sorted)
        for i, (xi, y_et, y_event) in enumerate(zip(self.X_sorted, self.y_ets_sorted, self.y_events_sorted)):
            Xjs = self.X_sorted[:max(i - 1, 0)]
            if i % 100 == 0:
                self.logger.debug(f'x{i}/x{len(self.X_sorted)}, S(t) = {st.item():.3f}')
            num = self.W(xp=xp, xj=xi, norm=norm)
            Xjs_w_sum = sum(np.array([self.W(xp=xp, xj=xj, norm=norm) for xj in Xjs]))
            denum = 1 - Xjs_w_sum
            st += np.log(1 - num / denum + self.log_epsilon) if y_event else 0

            if y_et == self.event_times_[curr_req_i]:
                self.logger.debug(f'S({self.y_ets_sorted[curr_req_i]}) = {st.item():.3f} by x{len(Xjs)}')
                st_horizon[curr_req_i] = st
                curr_req_i += 1
                if curr_req_i == len(self.event_times_):
                    return st_horizon

        return st_horizon

    def StxLogSeria(self, xp: np.ndarray) -> np.ndarray:
        # data should be normalized
        for fi in range(self.X_sorted.shape[1]):
            assert 0 <= xp[fi] <= 1, f"xp[{fi}] = {xp[fi]}"

        st = np.zeros(1)
        st_horizon = np.zeros(len(self.event_times_))

        norm = sum(np.array([self.K(xi=xp, xj=xk) for xk in self.X_sorted]))
        curr_req_i = 0

        assert len(self.X_sorted) == len(self.y_ets_sorted)
        for i, (xi, y_et, y_event) in enumerate(zip(self.X_sorted, self.y_ets_sorted, self.y_events_sorted)):
            Xjs = self.X_sorted[:max(i - 1, 0)]
            if i % 100 == 0:
                self.logger.debug(f'x{i}/x{len(self.X_sorted)}, S(t) = {st.item():.3f}')
            curr_K = self.W(xp=xp, xj=xi, norm=norm)
            Xjs_w_sum = sum(np.array([self.W(xp=xp, xj=xj, norm=norm) for xj in Xjs]))
            st -= y_event * curr_K * (1 - 0.5 * curr_K + Xjs_w_sum)

            if y_et == self.event_times_[curr_req_i]:
                self.logger.debug(f'S({self.y_ets_sorted[curr_req_i]}) = {st.item():.3f} by x{len(Xjs)}')
                st_horizon[curr_req_i] = st
                curr_req_i += 1
                if curr_req_i == len(self.event_times_):
                    return st_horizon

        return st_horizon

    def _predict_optimized(self, xps: np.ndarray, st_fn) -> np.ndarray:
        y_pred = np.zeros((len(xps), len(self.event_times_)))
        for i, xp in enumerate(xps):
            y_pred[i] = st_fn(xp=xp)

        return y_pred

    def predict_survival_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_optimized(xps=xps, st_fn=self.Stx)

    def predict_log_survival_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_optimized(xps=xps, st_fn=self.StxLog)

    def predict_log_survival_seria_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_optimized(xps=xps, st_fn=self.StxLogSeria)

    ####################################################################################################################
    # ------------------------------------------------ TORCH IMPLEMENTATIONS -------------------------------------------
    ####################################################################################################################
    def _StxTorchOnKernels(self, kernel_preds):
        w_cumsum = torch.cumsum(kernel_preds, dim=1)
        shifted_w_cumsum = w_cumsum - kernel_preds
        ones = torch.ones_like(shifted_w_cumsum)
        anomaly_mask = torch.isclose(shifted_w_cumsum, ones) | torch.isclose(w_cumsum, ones)
        shifted_w_cumsum[anomaly_mask] = 0.0
        w_cumsum[anomaly_mask] = 0.0

        xi = torch.log(1.0 - w_cumsum) - torch.log(1.0 - shifted_w_cumsum)

        filtered_xi = self.y_events_sorted_torch.unsqueeze(0).unsqueeze(-1) * xi

        hazards = torch.cumsum(filtered_xi, dim=1)
        return torch.exp(hazards)

    def _StxLogTorchOnKernels(self, kernel_preds):
        w_cumsum = torch.cumsum(kernel_preds, dim=1)
        shifted_w_cumsum = w_cumsum - kernel_preds
        ones = torch.ones_like(shifted_w_cumsum)
        anomaly_mask = torch.isclose(shifted_w_cumsum, ones) | torch.isclose(w_cumsum, ones)
        shifted_w_cumsum[anomaly_mask] = 0.0
        w_cumsum[anomaly_mask] = 0.0

        xi = torch.log(1.0 - w_cumsum) - torch.log(1.0 - shifted_w_cumsum)

        filtered_xi = self.y_events_sorted_torch.unsqueeze(0).unsqueeze(-1) * xi

        log_surv = torch.cumsum(filtered_xi, dim=1)
        return log_surv

    def _StxLogSeriaTorchOnKernels(self, kernel_preds):
        w_cumsum = torch.cumsum(kernel_preds, dim=1)
        anomaly_mask = torch.isclose(w_cumsum, torch.ones_like(w_cumsum))
        w_cumsum[anomaly_mask] = 0.0
        xi = -kernel_preds * (1 - 0.5 * kernel_preds + w_cumsum)
        filtered_xi = self.y_events_sorted_torch.unsqueeze(0).unsqueeze(-1) * xi
        log_surv = torch.cumsum(filtered_xi, dim=1)
        return log_surv

    def _StxSurvSeriaTorchOnKernels(self, kernel_preds):
        return torch.exp(self._StxLogSeriaTorchOnKernels(kernel_preds))

    def _predict_torch_optimized(self, xps: np.ndarray, st_fn) -> np.ndarray:
        if isinstance(xps, pd.DataFrame):
            xps = xps.to_numpy()
        xps_torch = torch.tensor(xps)

        b_norm = self.b
        b_torch = torch.tensor(b_norm)

        # slow legacy
        # kernel_preds = torch.tensor(
        #     [[self.kernel_fn(xi=xp, xj=xk, b=b_torch) for xk in self.X_sorted_torch] for xp in xps_torch]
        # )

        # speed up
        xps_torch_rep = xps_torch.repeat_interleave(len(self.X_sorted_torch), 0)
        x_train_rep = self.X_sorted_torch.repeat(len(xps_torch), 1)
        b_torch_rep = b_torch[None].repeat(len(xps_torch_rep), 1)
        kernel_preds = self.kernel_width * b_torch_rep * self.kernel_dist_fn(xps_torch_rep - x_train_rep)
        kernel_preds = torch.exp(-torch.mean(kernel_preds, axis=-1)) \
            .reshape(len(xps), len(self.X_sorted_torch))

        kernel_preds /= kernel_preds.sum(axis=1)[:, None]
        S = st_fn(kernel_preds[:, :, None])
        if len(self.ids_with_repeats) != len(self.y_ets_sorted):
            return S.detach().numpy()[:, :, 0][:, self.ids_with_repeats]
        else:
            return S.detach().numpy()[:, :, 0]

    def predict_survival_torch_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_torch_optimized(xps=xps, st_fn=self._StxTorchOnKernels)

    def predict_log_torch_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_torch_optimized(xps=xps, st_fn=self._StxLogTorchOnKernels)

    def predict_log_seria_torch_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_torch_optimized(xps=xps, st_fn=self._StxLogSeriaTorchOnKernels)

    def predict_surv_seria_torch_optimized(self, xps: np.ndarray) -> np.ndarray:
        return self._predict_torch_optimized(xps=xps, st_fn=self._StxSurvSeriaTorchOnKernels)
