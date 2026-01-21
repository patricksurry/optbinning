"""
Binning transformations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers

import numpy as np
import pandas as pd

from sklearn.utils import check_array

from .binning_statistics import bin_categorical
from .binning_statistics import bin_str_format


def transform_event_rate_to_woe(event_rate, n_nonevent, n_event):
    """Transform event rate to WoE.

    Parameters
    ----------
    event_rate : array-like or float
        Event rate.

    n_nonevent : int
        Total number of non-events.

    n_event : int
        Total number of events.

    Returns
    -------
    woe : numpy.ndarray or float
        Weight of evidence.
    """
    return np.log((1. / event_rate - 1) * n_event / n_nonevent)


def transform_woe_to_event_rate(woe, n_nonevent, n_event):
    """Transform WoE to event rate.

    Parameters
    ----------
    woe : array-like or float
        Weight of evidence.

    n_nonevent : int
        Total number of non-events.

    n_event : int
        Total number of events.

    Returns
    -------
    event_rate : numpy.ndarray or float
        Event rate.
    """
    return 1.0 / (1.0 + n_nonevent / n_event * np.exp(woe))


def _check_metric_special_missing(metric_special, metric_missing):
    if isinstance(metric_special, str):
        if metric_special != "empirical":
            raise ValueError('Invalid value for metric_special. Allowed '
                             'value "empirical"; got {}.'
                             .format(metric_special))

    elif isinstance(metric_special, dict):
        for k, v in metric_special.items():
            if not isinstance(v, numbers.Number):
                raise ValueError('Invalid value for metric_special key: {}.'
                                 .format(k))

    elif not isinstance(metric_special, numbers.Number):
        raise ValueError('Invalid value for metric_special. Allowed values '
                         'are "empirical" a numeric value or a dict; got {}.'
                         .format(metric_special))

    if isinstance(metric_missing, str):
        if metric_missing != "empirical":
            raise ValueError('Invalid value for metric_missing. Allowed '
                             'value "empirical"; got {}.'
                             .format(metric_missing))

    elif not isinstance(metric_missing, numbers.Number):
        raise ValueError('Invalid value for metric_missing. Allowed values '
                         'are "empirical" or a numeric value; got {}.'
                         .format(metric_missing))


def _check_show_digits(show_digits):
    if (not isinstance(show_digits, numbers.Integral) or
            not 0 <= show_digits <= 8):
        raise ValueError("show_digits must be an integer in [0, 8]; "
                         "got {}.".format(show_digits))


def _check_cat_unknown(metric, cat_unknown):
    if cat_unknown is not None:
        if not isinstance(cat_unknown, str) and metric == "bins":
            raise ValueError("Invalid value for cat_unknown. cat_unknown "
                             "must be string if metric='bins'.")

        if not isinstance(cat_unknown, int) and metric == "indices":
            raise ValueError("Invalid value for cat_unknown. cat_unknown "
                             "must be an integer if metric='indices'.")

        if metric in ("woe", "event_rate", "mean"):
            if not isinstance(cat_unknown, numbers.Number):
                raise ValueError("Invalid value for cat_unknown. cat_unknown "
                                 "must be numeric if metric='{}'."
                                 .format(metric))


def _retrieve_special_codes(special_codes):
    _special_codes = []
    for s in special_codes.values():
        if isinstance(s, (list, np.ndarray)):
            _special_codes.extend(s)
        else:
            _special_codes.append(s)

    return _special_codes


def _mask_special_missing(x, special_codes):
    if np.issubdtype(x.dtype, np.number):
        missing_mask = np.isnan(x)
    else:
        missing_mask = pd.isnull(x)

    if special_codes is None:
        special_mask = None
        n_special = 1
        clean_mask = ~missing_mask
    else:
        if isinstance(special_codes, dict):
            n_special = len(special_codes)
            _special_codes = _retrieve_special_codes(special_codes)
        else:
            n_special = 1
            _special_codes = special_codes

        special_mask = pd.Series(x).isin(_special_codes).values
        clean_mask = ~missing_mask & ~special_mask

    return special_mask, missing_mask, clean_mask, n_special


def _transform_metric_indices_bins(x, special_codes, metric, n_bins,
                                   n_special, bins_str, cat_unknown):

    if cat_unknown is None:
        if metric == 'indices':
            cat_unknown = -1
        else:
            cat_unknown = 'unknown'

    if metric == "indices":
        metric_value = np.arange(n_bins + n_special + 1)
        x_transform = np.full(x.shape, cat_unknown, dtype=int)
    elif metric == "bins":
        if isinstance(special_codes, dict):
            bins_str.extend(list(special_codes) + ["Missing"])
        else:
            bins_str.extend(["Special", "Missing"])

        metric_value = bins_str
        x_transform = np.full(x.shape, cat_unknown, dtype=object)

    return metric_value, x_transform


def _apply_special_missing(x, special_codes, metric, metric_special,
                           metric_missing, metric_value, special_mask,
                           missing_mask, x_transform, n_bins, n_special):
    """Apply special codes and missing value transformations.

    This handles the special codes and missing values after the main
    binning transformation has been applied to clean data.
    """
    if special_codes:
        if isinstance(special_codes, dict):
            xt = pd.Series(x)
            for i, (k, s) in enumerate(special_codes.items()):
                sl = s if isinstance(s, (list, np.ndarray)) else [s]
                mask = xt.isin(sl).values
                if (metric_special == "empirical" or (metric == "indices" and
                    not isinstance(metric_special, int)) or
                        metric == "bins"):
                    x_transform[mask] = metric_value[n_bins + i]
                else:
                    x_transform[mask] = metric_special
        else:
            if (metric_special == "empirical" or
                (metric == "indices" and
                    not isinstance(metric_special, int)) or
                    metric == "bins"):
                x_transform[special_mask] = metric_value[n_bins]
            else:
                x_transform[special_mask] = metric_special

    if (metric_missing == "empirical" or
        (metric == "indices" and not isinstance(metric_missing, int)) or
            metric == "bins"):
        x_transform[missing_mask] = metric_value[n_bins + n_special]
    else:
        x_transform[missing_mask] = metric_missing

    return x_transform


def _apply_transform(x, dtype, special_codes, metric, metric_special,
                     metric_missing, metric_value, clean_mask, special_mask,
                     missing_mask, indices, x_transform, x_clean, bins, n_bins,
                     n_special, cat_unknown):
    """Legacy transform function that applies binning and special/missing values.

    This function is used by the direct transform_*_target() functions.
    The cached transformer classes use a more optimized path.
    """
    # Apply binning to clean data
    if dtype == "numerical":
        # For bins metric, metric_value is a list of strings, so we need to convert
        # to array first to support fancy indexing
        if isinstance(metric_value, list):
            metric_value_array = np.array(metric_value, dtype=object)
            x_transform[clean_mask] = metric_value_array[indices.astype(int)]
        else:
            x_transform[clean_mask] = metric_value[indices.astype(int)]
    else:
        x_p = pd.Series(x)
        for i in range(n_bins):
            mask = x_p.isin(bins[i])
            x_transform[mask] = metric_value[i]

    # Apply special and missing values
    x_transform = _apply_special_missing(
        x, special_codes, metric, metric_special, metric_missing,
        metric_value, special_mask, missing_mask, x_transform,
        n_bins, n_special)

    return x_transform


class BinaryTargetTransformer:
    """Pre-computed transformer for binary target binning.

    This class caches all data-independent computations from transform_binary_target,
    allowing for faster repeated transforms with the same parameters.

    Parameters
    ----------
    splits : array-like
        Split points for numerical variables or categories for categorical variables.

    dtype : str
        Variable type: "numerical" or "categorical".

    n_nonevent : array-like
        Number of non-events per bin (including special/missing if empirical).

    n_event : array-like
        Number of events per bin (including special/missing if empirical).

    special_codes : array-like, dict or None
        Special codes to handle separately.

    categories : array-like or None
        Categories for categorical variables.

    cat_others : array-like or None
        Categories grouped as "Others".

    cat_unknown : float, int, str or None
        Value to assign to unknown categories.

    user_splits : array-like or None
        User-provided split points.

    metric : str
        Transform metric: "woe", "event_rate", "indices", or "bins".

    metric_special : float, str or dict
        Metric value for special codes.

    metric_missing : float or str
        Metric value for missing values.

    show_digits : int
        Significant digits for bin string formatting.
    """

    def __init__(self, splits, dtype, n_nonevent, n_event, special_codes,
                 categories, cat_others, cat_unknown, user_splits,
                 metric, metric_special, metric_missing, show_digits):

        # Validate parameters
        if metric not in ("event_rate", "woe", "indices", "bins"):
            raise ValueError('Invalid value for metric. Allowed string '
                           'values are "event_rate", "woe", "indices" and '
                           '"bins".')

        _check_cat_unknown(metric, cat_unknown)
        _check_metric_special_missing(metric_special, metric_missing)
        _check_show_digits(show_digits)

        self.splits = splits
        self.dtype = dtype
        self.special_codes = special_codes
        self.categories = categories
        self.cat_others = cat_others
        self.user_splits = user_splits
        self.metric = metric
        self.metric_special = metric_special
        self.metric_missing = metric_missing

        # Pre-compute bins and n_special
        if dtype == "numerical":
            self.bins = np.concatenate([[-np.inf], splits, [np.inf]])
            self.bins_str = bin_str_format(self.bins, show_digits) if metric == 'bins' else []
            self.n_bins = len(splits) + 1
        else:
            self.bins = bin_categorical(splits, categories, cat_others, user_splits)
            self.bins_str = [str(b) for b in self.bins] if metric == 'bins' else []
            self.n_bins = len(self.bins)

        # Pre-compute n_special
        if special_codes is None:
            self.n_special = 1
        elif isinstance(special_codes, dict):
            self.n_special = len(special_codes)
        else:
            self.n_special = 1

        # Pre-compute metric values
        if metric in ("woe", "event_rate"):
            self.metric_value, self.cat_unknown = self._precompute_woe_event_rate(
                n_nonevent, n_event, metric, metric_special, metric_missing, cat_unknown)
        else:
            self.metric_value, self.cat_unknown = self._precompute_indices_bins(
                metric, cat_unknown)

        # Pre-compute category-to-value mapping for categorical variables
        if dtype == "categorical":
            self.category_map = {}
            for i, bin_categories in enumerate(self.bins):
                for cat in bin_categories:
                    self.category_map[cat] = self.metric_value[i]
        else:
            self.category_map = None

    def _precompute_woe_event_rate(self, n_nonevent, n_event, metric,
                                    metric_special, metric_missing, cat_unknown):
        """Pre-compute WoE or event_rate arrays."""
        n_records = n_event + n_nonevent
        t_n_nonevent = n_nonevent.sum()
        t_n_event = n_event.sum()

        if "empirical" not in (metric_special, metric_missing):
            n_event = n_event[:self.n_bins]
            n_nonevent = n_nonevent[:self.n_bins]
            n_records = n_records[:self.n_bins]

        # Compute event rate and WoE
        mask = (n_event > 0) & (n_nonevent > 0)
        event_rate = np.zeros(len(n_records))
        woe = np.zeros(len(n_records))

        event_rate[mask] = n_event[mask] / n_records[mask]
        woe[mask] = transform_event_rate_to_woe(event_rate[mask], t_n_nonevent, t_n_event)

        # Compute default cat_unknown
        if cat_unknown is None:
            mean_event_rate = t_n_event / n_records.sum()
            if metric == "woe":
                cat_unknown = transform_event_rate_to_woe(mean_event_rate, t_n_nonevent, t_n_event)
            else:
                cat_unknown = mean_event_rate

        metric_value = woe if metric == "woe" else event_rate
        return metric_value, cat_unknown

    def _precompute_indices_bins(self, metric, cat_unknown):
        """Pre-compute indices or bins arrays."""
        if cat_unknown is None:
            cat_unknown = -1 if metric == 'indices' else 'unknown'

        if metric == "indices":
            metric_value = np.arange(self.n_bins + self.n_special + 1)
        elif metric == "bins":
            if isinstance(self.special_codes, dict):
                metric_value = self.bins_str + list(self.special_codes) + ["Missing"]
            else:
                metric_value = self.bins_str + ["Special", "Missing"]
        else:
            raise ValueError(f'Invalid metric for indices/bins: {metric}')

        return metric_value, cat_unknown

    def transform(self, x):
        """Transform data using pre-computed values.

        Parameters
        ----------
        x : array-like
            Input data to transform.

        Returns
        -------
        x_transform : numpy.ndarray
            Transformed data.
        """
        x = np.asarray(x)

        # Data-dependent: create masks
        special_mask, missing_mask, clean_mask, _ = _mask_special_missing(x, self.special_codes)
        x_clean = x[clean_mask]

        # Initialize output with pre-computed cat_unknown
        if self.metric == "indices":
            x_transform = np.full(x.shape, self.cat_unknown, dtype=int)
        elif self.metric == "bins":
            x_transform = np.full(x.shape, self.cat_unknown, dtype=object)
        else:
            x_transform = np.full(x.shape, self.cat_unknown)

        # Data-dependent: bin assignment and value mapping
        if self.dtype == "numerical":
            if len(self.splits):
                indices = np.digitize(x_clean, self.splits, right=False)
            else:
                indices = np.zeros(x_clean.shape, dtype=int)

            # Use vectorized indexing for numerical data
            if isinstance(self.metric_value, list):
                metric_value_array = np.array(self.metric_value, dtype=object)
                x_transform[clean_mask] = metric_value_array[indices.astype(int)]
            else:
                x_transform[clean_mask] = self.metric_value[indices.astype(int)]
        else:
            # Use pre-computed mapping for categorical data (avoids loop with .isin())
            if self.category_map is not None:
                x_transform[clean_mask] = pd.Series(x_clean).map(self.category_map).fillna(self.cat_unknown).values

        # Apply pre-computed special and missing values
        x_transform = _apply_special_missing(
            x, self.special_codes, self.metric, self.metric_special,
            self.metric_missing, self.metric_value, special_mask,
            missing_mask, x_transform, self.n_bins, self.n_special)

        return x_transform


def transform_binary_target(splits, dtype, x, n_nonevent, n_event,
                            special_codes, categories, cat_others, cat_unknown,
                            metric, metric_special, metric_missing,
                            user_splits, show_digits, check_input=False):

    if metric not in ("event_rate", "woe", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                        'values are "event_rate", "woe", "indices" and '
                        '"bins".')

    _check_cat_unknown(metric, cat_unknown)
    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)

    special_mask, missing_mask, clean_mask, n_special = _mask_special_missing(
        x, special_codes)

    x_clean = x[clean_mask]

    if dtype == "numerical":
        if len(splits):
            indices = np.digitize(x_clean, splits, right=False)
        else:
            indices = np.zeros(x_clean.shape)

        bins = np.concatenate([[-np.inf], splits, [np.inf]])
        bins_str = bin_str_format(bins, show_digits) if 'metric' == 'bins' else []
        n_bins = len(splits) + 1
    else:
        indices = None
        bins = bin_categorical(splits, categories, cat_others, user_splits)
        bins_str = [str(b) for b in bins] if 'metric' == 'bins' else []
        n_bins = len(bins)

    if metric in ("woe", "event_rate"):
        # Compute event rate and WoE
        n_records = n_event + n_nonevent
        t_n_nonevent = n_nonevent.sum()
        t_n_event = n_event.sum()

        if "empirical" not in (metric_special, metric_missing):
            n_event = n_event[:n_bins]
            n_nonevent = n_nonevent[:n_bins]
            n_records = n_records[:n_bins]

        # Default woe and event rate is 0
        mask = (n_event > 0) & (n_nonevent > 0)
        event_rate = np.zeros(len(n_records))
        woe = np.zeros(len(n_records))

        event_rate[mask] = n_event[mask] / n_records[mask]
        woe[mask] = transform_event_rate_to_woe(
            event_rate[mask], t_n_nonevent, t_n_event)

        # Assign unknown category value
        if cat_unknown is None:
            mean_event_rate = t_n_event / n_records.sum()
            if metric == "woe":
                cat_unknown = transform_event_rate_to_woe(
                    mean_event_rate, t_n_nonevent, t_n_event)
            else:
                cat_unknown = mean_event_rate

        # Assign normal metric values
        if metric == "woe":
            metric_value = woe
        else:
            metric_value = event_rate

        x_transform = np.full(x.shape, cat_unknown)
    else:
        # Assign corresponding indices or bin intervals
        metric_value, x_transform = _transform_metric_indices_bins(
            x, special_codes, metric, n_bins, n_special, bins_str, cat_unknown)

    x_transform = _apply_transform(
        x, dtype, special_codes, metric, metric_special, metric_missing,
        metric_value, clean_mask, special_mask, missing_mask, indices,
        x_transform, x_clean, bins, n_bins, n_special, cat_unknown)

    return x_transform


def transform_multiclass_target(splits, x, n_event, special_codes, metric,
                                metric_special, metric_missing, show_digits,
                                check_input=False):

    if metric not in ("mean_woe", "weighted_mean_woe", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                            'values are "mean_woe", "weighted_mean_woe", '
                            '"indices" and "bins".')

    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)

    special_mask, missing_mask, clean_mask, n_special = _mask_special_missing(
        x, special_codes)

    x_clean = x[clean_mask]

    if len(splits):
        indices = np.digitize(x_clean, splits, right=False)
    else:
        indices = np.zeros(x_clean.shape)

    bins = np.concatenate([[-np.inf], splits, [np.inf]])
    bins_str = bin_str_format(bins, show_digits) if metric == 'bins' else []
    n_bins = len(splits) + 1

    if metric in ("mean_woe", "weighted_mean_woe"):
        # Build non-event to compute one-vs-all WoE
        n_classes = n_event.shape[1]
        n_records = np.tile(n_event.sum(axis=1), (n_classes, 1)).T
        n_nonevent = n_records - n_event
        t_n_nonevent = n_nonevent.sum(axis=0)
        t_n_event = n_event.sum(axis=0)

        mask = (n_event > 0) & (n_nonevent > 0)
        event_rate = np.zeros(n_event.shape)
        woe = np.zeros(n_event.shape)

        event_rate[mask] = n_event[mask] / n_records[mask]

        for i in range(n_classes):
            woe[mask[:, i],  i] = transform_event_rate_to_woe(
                event_rate[mask[:, i], i], t_n_nonevent[i], t_n_event[i])

        if metric == "mean_woe":
            metric_value = woe.mean(axis=1)
        elif metric == "weighted_mean_woe":
            metric_value = np.average(woe, weights=t_n_event, axis=1)

        x_transform = np.zeros(x.shape)
    else:
        # Assign corresponding indices or bin intervals
        metric_value, x_transform = _transform_metric_indices_bins(
            x, special_codes, metric, n_bins, n_special, bins_str, None)

    x_transform = _apply_transform(
        x, "numerical", special_codes, metric, metric_special, metric_missing,
        metric_value, clean_mask, special_mask, missing_mask, indices,
        x_transform, x_clean, bins, n_bins, n_special, None)

    return x_transform


class ContinuousTargetTransformer:
    """Pre-computed transformer for continuous target binning.

    This class caches all data-independent computations from transform_continuous_target,
    allowing for faster repeated transforms with the same parameters.

    Parameters
    ----------
    splits : array-like
        Split points for numerical variables or categories for categorical variables.

    dtype : str
        Variable type: "numerical" or "categorical".

    n_records : array-like
        Number of records per bin (including special/missing if empirical).

    sums : array-like
        Sum of target values per bin (including special/missing if empirical).

    special_codes : array-like, dict or None
        Special codes to handle separately.

    categories : array-like or None
        Categories for categorical variables.

    cat_others : array-like or None
        Categories grouped as "Others".

    cat_unknown : float, int, str or None
        Value to assign to unknown categories.

    user_splits : array-like or None
        User-provided split points.

    metric : str
        Transform metric: "mean", "indices", or "bins".

    metric_special : float, str or dict
        Metric value for special codes.

    metric_missing : float or str
        Metric value for missing values.

    show_digits : int
        Significant digits for bin string formatting.
    """

    def __init__(self, splits, dtype, n_records, sums, special_codes,
                 categories, cat_others, cat_unknown, user_splits,
                 metric, metric_special, metric_missing, show_digits):

        # Validate parameters
        if metric not in ("mean", "indices", "bins"):
            raise ValueError('Invalid value for metric. Allowed string '
                           'values are "mean", "indices" and "bins".')

        _check_cat_unknown(metric, cat_unknown)
        _check_metric_special_missing(metric_special, metric_missing)
        _check_show_digits(show_digits)

        self.splits = splits
        self.dtype = dtype
        self.special_codes = special_codes
        self.categories = categories
        self.cat_others = cat_others
        self.user_splits = user_splits
        self.metric = metric
        self.metric_special = metric_special
        self.metric_missing = metric_missing

        # Pre-compute bins and n_special
        if dtype == "numerical":
            self.bins = np.concatenate([[-np.inf], splits, [np.inf]])
            self.bins_str = bin_str_format(self.bins, show_digits) if metric == 'bins' else []
            self.n_bins = len(splits) + 1
        else:
            self.bins = bin_categorical(splits, categories, cat_others, user_splits)
            self.bins_str = [str(b) for b in self.bins] if metric == 'bins' else []
            self.n_bins = len(self.bins)

        # Pre-compute n_special
        if special_codes is None:
            self.n_special = 1
        elif isinstance(special_codes, dict):
            self.n_special = len(special_codes)
        else:
            self.n_special = 1

        # Pre-compute metric values
        if metric == "mean":
            self.metric_value, self.cat_unknown = self._precompute_mean(
                n_records, sums, metric_special, metric_missing, cat_unknown)
        else:
            self.metric_value, self.cat_unknown = self._precompute_indices_bins(
                metric, cat_unknown)

        # Pre-compute category-to-value mapping for categorical variables
        if dtype == "categorical":
            self.category_map = {}
            for i, bin_categories in enumerate(self.bins):
                for cat in bin_categories:
                    self.category_map[cat] = self.metric_value[i]
        else:
            self.category_map = None

    def _precompute_mean(self, n_records, sums, metric_special, metric_missing, cat_unknown):
        """Pre-compute mean arrays."""
        if "empirical" not in (metric_special, metric_missing):
            n_records = n_records[:self.n_bins]
            sums = sums[:self.n_bins]

        # Compute mean
        mask = n_records > 0
        metric_value = np.zeros(len(n_records))
        metric_value[mask] = sums[mask] / n_records[mask]

        # Compute default cat_unknown
        if cat_unknown is None:
            cat_unknown = sums.sum() / n_records.sum()

        return metric_value, cat_unknown

    def _precompute_indices_bins(self, metric, cat_unknown):
        """Pre-compute indices or bins arrays."""
        if cat_unknown is None:
            cat_unknown = -1 if metric == 'indices' else 'unknown'

        if metric == "indices":
            metric_value = np.arange(self.n_bins + self.n_special + 1)
        elif metric == "bins":
            if isinstance(self.special_codes, dict):
                metric_value = self.bins_str + list(self.special_codes) + ["Missing"]
            else:
                metric_value = self.bins_str + ["Special", "Missing"]
        else:
            raise ValueError(f'Invalid metric for indices/bins: {metric}')

        return metric_value, cat_unknown

    def transform(self, x):
        """Transform data using pre-computed values.

        Parameters
        ----------
        x : array-like
            Input data to transform.

        Returns
        -------
        x_transform : numpy.ndarray
            Transformed data.
        """
        x = np.asarray(x)

        # Data-dependent: create masks
        special_mask, missing_mask, clean_mask, _ = _mask_special_missing(x, self.special_codes)
        x_clean = x[clean_mask]

        # Initialize output with pre-computed cat_unknown
        if self.metric == "indices":
            x_transform = np.full(x.shape, self.cat_unknown, dtype=int)
        elif self.metric == "bins":
            x_transform = np.full(x.shape, self.cat_unknown, dtype=object)
        else:
            x_transform = np.full(x.shape, self.cat_unknown)

        # Data-dependent: bin assignment and value mapping
        if self.dtype == "numerical":
            if len(self.splits):
                indices = np.digitize(x_clean, self.splits, right=False)
            else:
                indices = np.zeros(x_clean.shape, dtype=int)

            # Use vectorized indexing for numerical data
            if isinstance(self.metric_value, list):
                metric_value_array = np.array(self.metric_value, dtype=object)
                x_transform[clean_mask] = metric_value_array[indices.astype(int)]
            else:
                x_transform[clean_mask] = self.metric_value[indices.astype(int)]
        else:
            # Use pre-computed mapping for categorical data (avoids loop with .isin())
            if self.category_map is not None:
                x_transform[clean_mask] = pd.Series(x_clean).map(self.category_map).fillna(self.cat_unknown).values

        # Apply pre-computed special and missing values
        x_transform = _apply_special_missing(
            x, self.special_codes, self.metric, self.metric_special,
            self.metric_missing, self.metric_value, special_mask,
            missing_mask, x_transform, self.n_bins, self.n_special)

        return x_transform


def transform_continuous_target(splits, dtype, x, n_records, sums,
                                special_codes, categories, cat_others,
                                cat_unknown, metric, metric_special,
                                metric_missing, user_splits, show_digits,
                                check_input=False):

    if metric not in ("mean", "indices", "bins"):
        raise ValueError('Invalid value for metric. Allowed string '
                            'values are "mean", "indices" and "bins".')
    _check_cat_unknown(metric, cat_unknown)
    _check_metric_special_missing(metric_special, metric_missing)
    _check_show_digits(show_digits)

    if check_input:
        x = check_array(x, ensure_2d=False, dtype=None,
                        ensure_all_finite='allow-nan')

    x = np.asarray(x)

    special_mask, missing_mask, clean_mask, n_special = _mask_special_missing(
        x, special_codes)

    x_clean = x[clean_mask]
    if dtype == "numerical":
        if len(splits):
            indices = np.digitize(x_clean, splits, right=False)
        else:
            indices = np.zeros(x_clean.shape)

        bins = np.concatenate([[-np.inf], splits, [np.inf]])
        bins_str = bin_str_format(bins, show_digits) if metric == 'bins' else []
        n_bins = len(splits) + 1
    else:
        indices = None
        bins = bin_categorical(splits, categories, cat_others, user_splits)
        bins_str = [str(b) for b in bins] if metric == 'bins' else []
        n_bins = len(bins)

    if "empirical" not in (metric_special, metric_missing):
        n_records = n_records[:n_bins]
        sums = sums[:n_bins]

    if metric == "mean":
        # Compute mean
        mask = n_records > 0
        metric_value = np.zeros(len(n_records))
        metric_value[mask] = sums[mask] / n_records[mask]

        # Assign unknown category value
        if cat_unknown is None:
            cat_unknown = sums.sum() / n_records.sum()

        x_transform = np.full(x.shape, cat_unknown)
    else:
        # Assign corresponding indices or bin intervals
        metric_value, x_transform = _transform_metric_indices_bins(
            x, special_codes, metric, n_bins, n_special, bins_str, cat_unknown)

    x_transform = _apply_transform(
        x, dtype, special_codes, metric, metric_special, metric_missing,
        metric_value, clean_mask, special_mask, missing_mask, indices,
        x_transform, x_clean, bins, n_bins, n_special, cat_unknown)

    return x_transform
