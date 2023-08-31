from typing import Callable, Union, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sksurv.functions import StepFunction

from survbex.optimizers import GradientOptimizer
from survlimepy import SurvLimeExplainer
from survlimepy.utils.optimisation import OptFuncionMaker
from survlimepy.utils.neighbours_generator import NeighboursGenerator


class SurvBexExplainer(SurvLimeExplainer):
    """Look for the coefficient of a COX model."""

    def __init__(
            self,
            training_features: Union[np.ndarray, pd.DataFrame],
            training_events: Union[np.ndarray, pd.Series, List[Union[bool, float, int]]],
            training_times: Union[np.ndarray, pd.Series, List[Union[float, int]]],
            model_output_times: Optional[np.ndarray] = None,
            H0: Optional[Union[np.ndarray, List[float], StepFunction]] = None,
            kernel_width: Optional[float] = None,
            functional_norm: Union[float, str] = 2,
            random_state: Optional[int] = None
    ) -> None:
        """Init function.

        Args:
            training_features (Union[np.ndarray, pd.DataFrame]): data used to train the bb model.
            training_events (Union[np.ndarray, pd.Series, List[Union[bool, float, int]]]): training events indicator.
            training_times (Union[np.ndarray, pd.Series, List[Union[float, int]]]): training times to event.
            model_output_times (Optional[np.ndarray]): output times of the bb model.
            H0 (Optional[Union[np.ndarray, List[float], StepFunction]]): baseline cumulative hazard.
            kernel_width (Optional[List[float]]): width of the kernel to be used to generate the neighbours and to compute distances.
            functional_norm (Optional[Union[float, str]]): functional norm to calculate the distance between the Cox model and the black box model.
            random_state (Optional[int]): number to be used for random seeds.

        Returns:
            None.
        """
        self.random_state = check_random_state(random_state)
        self.training_features = training_features
        self.training_events = training_events
        self.training_times = training_times
        self.model_output_times = model_output_times
        self.computed_weights = None
        self.montecarlo_weights = None
        self.is_data_frame = isinstance(self.training_features, pd.DataFrame)
        self.is_np_array = isinstance(self.training_features, np.ndarray)
        if not (self.is_data_frame or self.is_np_array):
            raise TypeError(
                "training_features must be either a numpy array or a pandas DataFrame."
            )
        if self.is_data_frame:
            self.feature_names = self.training_features.columns
            self.training_features_np = training_features.to_numpy()
        else:
            self.feature_names = [
                f"feature_{i}" for i in range(self.training_features.shape[1])
            ]
            self.training_features_np = np.copy(training_features)
        self.H0 = H0
        self.num_individuals = self.training_features.shape[0]
        self.num_features = self.training_features.shape[1]

        if kernel_width is None:
            num_sigma_opt = 4
            den_sigma_opt = self.num_individuals * (self.num_features + 2)
            pow_sigma_opt = 1 / (self.num_features + 4)
            kernel_default = (num_sigma_opt / den_sigma_opt) ** pow_sigma_opt
            self.kernel_width = kernel_default
        else:
            self.kernel_width = kernel_width

        self.functional_norm = functional_norm

    def explain_instance(
            self,
            data_row: Union[List[float], np.ndarray, pd.Series],
            predict_fn: Callable,
            type_fn: Literal["survival", "cumulative"] = "cumulative",
            num_samples: int = 1000,
            num_val_samples: int = 1000,
            max_difference_time_allowed: Optional[float] = None,
            max_hazard_value_allowed: Optional[float] = None,
            verbose: bool = False,
            optimizer: Literal["convex", "gradient"] = "convex",
            grid_info_file: [Optional[str]] = None,
            max_iter: Optional[int] = None

    ) -> np.ndarray:
        """Generates explanations for a prediction.

        Args:
            data_row (Union[List[float], np.ndarray, pd.Series]): data point to be explained.
            predict_fn (Callable): function that computes cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            num_samples (int): number of neighbours to use.
            max_difference_time_allowed (Optional[float]): maximum difference between times allowed. If a difference exceeds this value, then max_difference_time_allowed will be used.
            max_hazard_value_allowed (Optional[float]): maximum hazard value allowed. If a prediction exceeds this value, then max_hazard_value_allows will be used.
            verbose (bool): whether or not to show cvxpy messages.
            optimizer (Literal["convex", "gradient"]): okay bro
        Returns:
            cox_values (np.ndarray): obtained weights from the convex problem.
        """
        # To be used while plotting
        if isinstance(data_row, list):
            self.data_point = np.array(data_row).reshape(1, -1)
        elif isinstance(data_row, np.ndarray):
            total_dimensions_data_row = len(data_row.shape)
            total_rows = data_row.shape[0]
            if total_dimensions_data_row == 1:
                self.data_point = np.reshape(data_row, newshape=(1, -1))
            elif total_dimensions_data_row == 2:
                if total_rows > 1:
                    raise ValueError("data_point must contain a single row.")
                self.data_point = data_row
            else:
                raise ValueError("data_point must not have more than 2 dimensions.")
        elif isinstance(data_row, pd.Series):
            self.data_point = data_row.to_numpy().reshape(1, -1)
        else:
            raise ValueError(
                "data_point must be a list, a numpy array or a pandas Series."
            )

        # Generate neighbours
        neighbours_generator = NeighboursGenerator(
            training_features=self.training_features_np,
            data_row=self.data_point,
            sigma=self.kernel_width,
            random_state=self.random_state,
        )

        neighbours = neighbours_generator.generate_neighbours(num_samples=num_samples)
        neighbours_transformed = self.transform_data(data=neighbours)

        # Solve optimisation problem
        args = dict(
            training_features=self.training_features_np,
            training_events=self.training_events,
            training_times=self.training_times,
            kernel_width=self.kernel_width,
            neighbours=neighbours,
            neighbours_transformed=neighbours_transformed,
            num_samples=num_samples,
            data_point=self.data_point,
            predict_fn=predict_fn,
            type_fn=type_fn,
            functional_norm=self.functional_norm,
            model_output_times=self.model_output_times,
            H0=self.H0,
            max_difference_time_allowed=max_difference_time_allowed,
            max_hazard_value_allowed=max_hazard_value_allowed,
            verbose=verbose,
        )
        if optimizer == 'convex':
            self.opt_funcion_maker = OptFuncionMaker(
                **args
            )
        elif optimizer == 'gradient':
            neighbours_val = neighbours_generator.generate_neighbours(num_samples=num_val_samples)
            neighbours_val_transformed = self.transform_data(data=neighbours_val)
            assert grid_info_file is not None, 'grid_info_file arg should be passed for BeX explainer'
            assert max_iter is not None, 'max_iter arg should be passed for BeX explainer'

            self.opt_funcion_maker = GradientOptimizer(
                **args,
                neighbours_val=neighbours_val,
                neighbours_val_transformed=neighbours_val_transformed,
                grid_info_file=grid_info_file,
                max_iter=max_iter
            )
        else:
            raise Exception(f'Undefined optimizer = {optimizer}')

        b = self.opt_funcion_maker.solve_problem()
        self.computed_weights = np.copy(b)
        return b
