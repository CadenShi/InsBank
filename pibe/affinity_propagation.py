"""Affinity Propagation clustering algorithm."""

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org

# License: BSD 3 clause

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

import warnings
from numbers import Integral, Real
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np
import math
import torch
import copy

from sklearn._config import config_context
from sklearn.base import BaseEstimator, ClusterMixin
# from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import euclidean_distances, pairwise_distances_argmin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import check_is_fitted

from tqdm import trange  
from tqdm import tqdm
import logging
import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


def _cos_distance(X, Y):
    X_norm = X / X.norm(dim=1, keepdim=True)
    Y_norm = Y / Y.norm(dim=1, keepdim=True)
    cosine_distance = 1-torch.mm(X_norm, Y_norm.t())
    return cosine_distance


def _eu_distance(X, Y):
    distances_part = torch.cdist(X, Y, p=2)
    return distances_part


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])
    if preference is None:
        return False
    return all_equal_preferences() and all_equal_similarities()


def _affinity_propagation_cpu(
    S,
    *,
    preference,
    convergence_iter,
    max_iter,
    damping,
    verbose,
    return_n_iter,
    random_state,
    R_history=None,
    alpha=0.5,
    lamb=0.95,
):
    """Main affinity propagation algorithm."""
    n_samples = S.shape[0]
    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn(
            "All samples have mutually equal similarities. "
            "Returning arbitrary cluster center(s)."
        )
        if preference.flat[0] > S.flat[n_samples - 1]:
            return (
                (np.arange(n_samples), np.arange(n_samples), 0)
                if return_n_iter
                else (np.arange(n_samples), np.arange(n_samples))
            )
        else:
            return (
                (np.array([0]), np.array([0] * n_samples), 0)
                if return_n_iter
                else (np.array([0]), np.array([0] * n_samples))
            )

    if R_history is not None:
        R_history = R_history.detach().cpu().numpy()
    # Place preference on the diagonal of S
    if preference is not None:
        S.flat[:: (n_samples + 1)] = preference
    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.standard_normal(size=(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)
    for it in tqdm(range(max_iter)):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        if R_history is not None:
            R = R*(1-alpha) + R_history*alpha
            alpha = alpha * lamb

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[:: n_samples + 1] = R.flat[:: n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_samples
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                # print("Converged after %d iterations." % it)
                break
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars

    if K > 0:
        if never_converged:
            warnings.warn(
                (
                    "Affinity propagation did not converge, this model "
                    "may return degenerate cluster centers and labels."
                ),
                ConvergenceWarning,
            )
        # c = np.argmax(S[:, I], axis=1)
        # c[I] = np.arange(K)  # Identify clusters
        # # Refine the final set of exemplars and clusters and return results
        # for k in range(K):
        #     ii = np.where(c == k)[0]
        #     j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
        #     I[k] = ii[j]

        # c = np.argmax(S[:, I], axis=1)
        # c[I] = np.arange(K)
        # labels = I[c]
        # # Reduce labels to a sorted, gapless, list
        # cluster_centers_indices = np.unique(labels)
        # labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn(
            (
                "Affinity propagation did not converge and this model "
                "will not have any cluster centers."
            ),
            ConvergenceWarning,
        )
        # labels = np.array([-1] * n_samples)
        # cluster_centers_indices = []

    if return_n_iter:
        return None, None, None, A, R
    else:
        return None, None, A, R


def _affinity_propagation(S, *, preference, convergence_iter, max_iter, damping, verbose, return_n_iter, random_state, R_history=None, alpha=0.5, lamb=0.95):
    """Main affinity propagation algorithm with PyTorch."""
    dtype = torch.float32 if S.shape[0] < 39000 else torch.float16
    # scheduler = LambdaScheduler(lamb)
    with torch.no_grad():
        S = torch.tensor(S, dtype=dtype, device='cuda')
        n_samples = S.shape[0]
        device = S.device 

        A = torch.zeros((n_samples, n_samples), dtype=dtype, device=device)
        R = torch.zeros((n_samples, n_samples), dtype=dtype, device=device)  # Initialize messages
        tmp = torch.zeros((n_samples, n_samples), dtype=dtype, device=device)

        if R_history is not None:
            R_history = R_history.to(dtype).cuda()

        S += (
            torch.finfo(S.dtype).eps * S + torch.finfo(S.dtype).tiny * 100
        ) * torch.tensor(random_state.standard_normal(size=(n_samples, n_samples)), dtype=dtype, device=device)

        e = torch.zeros((n_samples, convergence_iter), dtype=dtype, device=device)
        ind = torch.arange(n_samples)


        for it in tqdm(range(max_iter)):
            torch.add(A, S, out=tmp)
            Y, I = torch.max(tmp, dim=1)
            tmp[range(n_samples), I] = -torch.inf
            Y2, _ = torch.max(tmp, dim=1)

            torch.sub(S, Y[:, None], out=tmp)
            tmp[ind, I] = S[ind, I] - Y2

            tmp *= (1 - damping)
            R *= damping
            R += tmp

            if R_history is not None:
                R = R * (1 - alpha) + R_history * alpha
                alpha *= lamb

            torch.maximum(R, torch.tensor(0, dtype=R.dtype), out=tmp)
            
            # Manually set diagonal values
            diag_indices = torch.arange(n_samples)
            tmp[diag_indices, diag_indices] = R[diag_indices, diag_indices]
            
            tmp -= tmp.sum(dim=0)
            dA = tmp.diagonal().clone()
            tmp = tmp.clip_(min=0)
            tmp[diag_indices, diag_indices] = dA  # Set the diagonal back
            
            tmp *= (1 - damping)
            A *= damping
            A -= tmp
            
            if torch.isinf(A).any():
                if (A == float('-inf')).sum():
                    logging.info("-Inf detected in A")
                else:
                    logging.info("Inf detected in A")
                raise ValueError
            if torch.isinf(R).any():
                if (R == float('-inf')).sum():
                    logging.info("-Inf detected in R")
                else:
                    logging.info("Inf detected in R")
                raise ValueError

            E = (A.diagonal() + R.diagonal()) > 0
            e[:, it % convergence_iter] = E
            K = E.sum(dim=0)

            if it >= convergence_iter:
                se = e.sum(dim=1)
                unconverged = torch.sum((se == convergence_iter) | (se == 0)) != n_samples
                if (not unconverged and (K > 0)) or (it == max_iter):
                    never_converged = False
                    if verbose:
                        print(f"Converged after {it} iterations.")
                    break
        else:
            never_converged = True
            if verbose:
                print("Did not converge")
        del R_history
    # I = torch.nonzero(E).squeeze()
    # K = I.numel()

    A = A.cpu().to(torch.float32).numpy()
    R = R.cpu().to(torch.float32).numpy()

    if return_n_iter:
        return None, None, None, A, R
    else:
        return None, None, A, R


###############################################################################
# Public API


# @validate_params(
#     {
#         "S": ["array-like"],
#         "return_n_iter": ["boolean"],
#     },
#     prefer_skip_nested_validation=False,
# )
def affinity_propagation(
    S,
    *,
    preference=None,
    convergence_iter=15,
    max_iter=200,
    damping=0.5,
    copy=True,
    verbose=False,
    return_n_iter=False,
    random_state=None,
):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations.

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Returns
    -------
    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it will still return a arrays of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cluster import affinity_propagation
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> S = -euclidean_distances(X, squared=True)
    >>> cluster_centers_indices, labels = affinity_propagation(S, random_state=0)
    >>> cluster_centers_indices
    array([0, 3])
    >>> labels
    array([0, 0, 0, 1, 1, 1])
    """
    estimator = AffinityPropagation(
        damping=damping,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        copy=copy,
        preference=preference,
        affinity="precomputed",
        verbose=verbose,
        random_state=random_state,
    ).fit(S)

    if return_n_iter:
        return estimator.cluster_centers_indices_, estimator.labels_, estimator.n_iter_
    return estimator.cluster_centers_indices_, estimator.labels_


class AffinityPropagation(ClusterMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "damping": [Interval(Real, 0.5, 1.0, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "convergence_iter": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
        "preference": [
            "array-like",
            Interval(Real, None, None, closed="neither"),
            None,
        ],
        "affinity": [StrOptions({"euclidean", "precomputed"})],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        damping=0.5,
        max_iter=200,
        convergence_iter=15,
        copy=True,
        preference=None,
        affinity="euclidean",
        verbose=True,
        random_state=None,
        batch_size=50000,
        n_clusters=6000,
        alpha=0.5,
        lamb=0.95,
        gamma=1.0,
        mode='multiply',
        device='gpu',
        save_log=True,
    ):
        if not save_log:
            logging.basicConfig(level=logging.CRITICAL)
        else:
            logging.basicConfig(level=logging.DEBUG,  # 设置最低日志级别为DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        ### parameters added by pibe
        self.mode = mode
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.A = None
        self.R = None
        self.representative_scores = None
        self.quality_scores = None
        self.pool_indexes = None
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.iter_cnt = -1
        self.last_count = 0
        self.X_old = None
        self.device=device
        self.save_log = save_log
        logging.info(f"Device: {device}")

    def _more_tags(self):
        return {"pairwise": self.affinity == "precomputed"}

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, quality=None):
        if self.affinity == "precomputed":
            X = self._validate_data(X, copy=self.copy, force_writeable=True)
            self.affinity_matrix_ = X
        elif self.affinity == "cosine_similarity":
            logging.info("Affinity: Cosine Similarity\n")
            X = X.cuda()
            self.affinity_matrix_ = []
            left = 0
            step = 10000
            right = 10000
            while left < X.shape[0]:
                right = min(left+step, X.shape[0])
                self.affinity_matrix_.append(_cos_distance(X[left:right, :], X))
                left += step
            self.affinity_matrix_ = (-torch.cat(self.affinity_matrix_, dim=0).pow(1)).detach().cpu().numpy()
            X.detach().cpu()
        elif self.affinity == "euclidean":
            logging.info("Affinity: Euclidean\n")
            X = X.cuda()
            self.affinity_matrix_ = []
            left = 0
            step = 10000
            right = 10000
            while left < X.shape[0]:
                right = min(left+step, X.shape[0])
                self.affinity_matrix_.append(_eu_distance(X[left:right, :], X))
                left += step
            self.affinity_matrix_ = (-torch.cat(self.affinity_matrix_, dim=0).pow(1)).detach().cpu().numpy()
            # self.affinity_matrix_ = self.affinity_matrix_.numpy()
            X.detach().cpu()
        else:
            raise NotImplementedError

        if self.affinity_matrix_.shape[0] != self.affinity_matrix_.shape[1]:
            raise ValueError(
                "The matrix of similarities must be a square array. "
                f"Got {self.affinity_matrix_.shape} instead."
            )

        if self.preference is None:
            preference = np.median(self.affinity_matrix_)
        else:
            preference = self.preference

        np.fill_diagonal(self.affinity_matrix_[:, :], preference)
        preference = np.asarray(preference)

        random_state = check_random_state(self.random_state)

        if self.device == 'gpu':
            (
                self.cluster_centers_indices_,
                self.labels_,
                self.n_iter_,
                self.A,
                self.R
            ) = _affinity_propagation(
                self.affinity_matrix_,
                max_iter=self.max_iter,
                convergence_iter=self.convergence_iter,
                preference=preference,
                # preference=None,
                damping=self.damping,
                verbose=self.verbose,
                return_n_iter=True,
                random_state=random_state,
                alpha=self.alpha,
                lamb=self.lamb,
            )
        elif self.device == 'cpu':
            (
                self.cluster_centers_indices_,
                self.labels_,
                self.n_iter_,
                self.A,
                self.R
            ) = _affinity_propagation_cpu(
                self.affinity_matrix_,
                max_iter=self.max_iter,
                convergence_iter=self.convergence_iter,
                preference=preference,
                # preference=None,
                damping=self.damping,
                verbose=self.verbose,
                return_n_iter=True,
                random_state=random_state,
                alpha=self.alpha,
                lamb=self.lamb,
            )
        else:
            raise NotImplementedError


        tmp = self.A + self.R
        assert not np.isinf(tmp).any(), "Inf detected in Fitness\n"
        column_sum = tmp.sum(axis=0)
        row_sum = tmp.sum(axis=1)
        representation = (column_sum - row_sum + np.diag(tmp)) / tmp.shape[0]
        assert not np.isinf(representation).any(), "Inf detected in Representation\n"
        min_val = np.min(representation)
        max_val = np.max(representation)
        scaled_representation = (representation - min_val) / (max_val - min_val)
        score = scaled_representation
        if quality is not None:
            quality = np.array(quality)
            min_val = np.min(quality)
            # min_val = np.mean(quality)
            # min_val = np.percentile(quality, 25)
            max_val = np.max(quality)
            scaled_quality = (quality - min_val) / (max_val - min_val)
            if self.mode == 'addition':
                score = scaled_representation + self.gamma*scaled_quality
            elif self.mode == 'multiply':
                score = (1+scaled_representation) * np.power(1+scaled_quality, self.gamma)
            elif 'nonlinear' in self.mode:
                quantile = float(self.mode.split('nonlinear_')[-1])
                ratio_upper = np.quantile(scaled_quality, quantile)
                ratio_lower = np.quantile(scaled_quality, 0.3)

                mul_coe = 4 / (ratio_upper - ratio_lower)
                sub_coe = ratio_lower + 2 / mul_coe
                scaled_quality = (scaled_quality - sub_coe) * mul_coe
                scaled_quality =  1 / (1 + np.exp(-scaled_quality))

                score = (1+scaled_representation) * (1+scaled_quality)
            else:
                raise NotImplementedError

        _, self.cluster_centers_indices_  = torch.topk(torch.tensor(score), self.n_clusters)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].clone()

        self.pool_indexes = self.cluster_centers_indices_.detach().clone()
        self.representative_scores = torch.tensor(scaled_representation)[self.cluster_centers_indices_].detach().clone()
        self.overall_scores = torch.tensor(score)[self.cluster_centers_indices_].detach().clone()
        if quality is not None:
            self.quality_scores = torch.tensor(quality)[self.cluster_centers_indices_].detach().clone()

        tmp_save = {}
        tmp_save['representation'] = representation
        tmp_save['scaled_representation_scores'] = scaled_representation
        if quality is not None:
            tmp_save['quality'] = quality
            tmp_save['scaled_quality_scores'] = torch.tensor(scaled_quality)
        if self.save_log:
            os.makedirs(f'./ap_logs/1223_cleaned_5sets_{self.affinity}_{self.mode}_alpha_{self.alpha}_lamb_{self.lamb}_gamma_{self.gamma}', exist_ok=True)
            torch.save(tmp_save, f'./ap_logs/1223_cleaned_5sets_{self.affinity}_{self.mode}_alpha_{self.alpha}_lamb_{self.lamb}_gamma_{self.gamma}/{self.iter_cnt}.pth')
        
        return self

    def incremental_fit(self, X_new, quality=None):
        """Progressive InsBank Evolution based on Affinity Propagtion   

        Parameters
        ----------
        X_new: {torch.tensor} of shape (n_samples, embedding_size)
                New candidate data to evolve with current InsBank
        quality: {torch.tensor} of shape (n_sampels, )
                Quality scores of new candidate data
        """
        self.iter_cnt += 1
        cur_num = X_new.shape[0]

        if self.A is None or self.R is None:
            self.last_count += cur_num
            self.X_old = X_new.clone()
            del X_new
            return self.fit(self.X_old, None, quality)

        with torch.no_grad():
            similarities = []
            self.X_old = self.X_old.cuda()
            X_new = X_new.cuda()
            left = 0
            step = 10000
            while left < X_new.shape[0]:
                right = min(left+step, X_new.shape[0])
                similarities.append(_cos_distance(X_new[left:right, :], self.X_old))
                left += step
            similarities = torch.cat(similarities, dim=0).cuda()
            self.X_old = self.X_old.detach().cpu()
            X_new = X_new.cpu()
            weights = (similarities / similarities.sum(dim=1).unsqueeze(1))    # weights is stored in GPU
            similarities = similarities.cpu()
            del similarities
            
            self.R = torch.tensor(self.R).float()

            R_new = torch.zeros(self.pool_indexes.shape[0]+X_new.shape[0], self.pool_indexes.shape[0]+X_new.shape[0]).cuda()
            logging.info("Start Initialize Momentum Responsibility\n")

            R_new[:self.cluster_centers_indices_.shape[0], :self.cluster_centers_indices_.shape[0]] = self.R[self.cluster_centers_indices_][:,self.cluster_centers_indices_].cuda()
            pool_size = self.cluster_centers_indices_.shape[0]
            start_idx = self.cluster_centers_indices_.shape[0]
            end_idx = R_new.shape[0]

            R_new[start_idx:end_idx, :pool_size] += torch.matmul(weights, self.R[:, self.cluster_centers_indices_].cuda())
            R_new[:pool_size, start_idx:end_idx] += torch.matmul(self.R[self.cluster_centers_indices_, :].cuda(), weights.transpose(0, 1))
            weights = weights.cpu()
            del weights

            mask = torch.ones(R_new.shape[0], R_new.shape[0], dtype=torch.bool)
            mask[pool_size:, pool_size:] = False
            R_new[pool_size:, pool_size:] = R_new[mask].median()
            R_new = R_new.detach().cpu()

            logging.info("Start Initialize Affinity Responsibility\n")
            self.X_old = torch.cat([self.X_old[self.cluster_centers_indices_], X_new], dim=0)
            del X_new, self.R
            
            if self.affinity == "cosine_similarity":
                logging.info("Affinity: Cosine Similarity\n")
                self.X_old = self.X_old.cuda()
                self.affinity_matrix_ = []
                left = 0
                step = 10000
                right = 10000
                while left < self.X_old.shape[0]:
                    right = min(left+step, self.X_old.shape[0])
                    self.affinity_matrix_.append(_cos_distance(self.X_old[left:right, :], self.X_old))
                    left += step
                self.affinity_matrix_ = (-torch.cat(self.affinity_matrix_, dim=0).pow(1)).detach().cpu().numpy()
                self.X_old = self.X_old.detach().cpu()
            elif self.affinity == "euclidean":
                logging.info("Affinity: Euclidean\n")
                self.X_old = self.X_old.cuda()
                self.affinity_matrix_ = []
                left = 0
                step = 10000
                right = 10000
                while left < self.X_old.shape[0]:
                    right = min(left+step, self.X_old.shape[0])
                    self.affinity_matrix_.append(_eu_distance(self.X_old[left:right, :], self.X_old))
                    left += step
                self.affinity_matrix_ = (-torch.cat(self.affinity_matrix_, dim=0).pow(1)).detach().cpu().numpy()
                self.X_old = self.X_old.detach().cpu()
            else:
                raise NotImplementedError
            if self.save_log:
                torch.save(R_new, f'./ap_logs/1223_cleaned_5sets_{self.affinity}_{self.mode}_alpha_{self.alpha}_lamb_{self.lamb}_gamma_{self.gamma}/momentum_responsibility_{self.iter_cnt}.pth')
                logging.info("Finish Initialize Momentum Responsibility\n")

            if self.preference is not None:
                preference = self.preference
            else:
                preference = np.median(self.affinity_matrix_)

            logging.info("Start fill Similarity\n")
            np.fill_diagonal(self.affinity_matrix_[:, :], preference)
            preference = np.asarray(preference)
            logging.info("Finish Initialize Affinity Responsibility\n")

            random_state = check_random_state(self.random_state)

            if self.device == 'gpu':
                (
                    self.cluster_centers_indices_,
                    self.labels_,
                    self.n_iter_,
                    self.A,
                    self.R
                ) = _affinity_propagation(
                    self.affinity_matrix_,
                    max_iter=self.max_iter,
                    convergence_iter=self.convergence_iter,
                    preference=preference,
                    damping=self.damping,
                    verbose=self.verbose,
                    return_n_iter=True,
                    random_state=random_state,
                    R_history=R_new,
                    alpha=self.alpha,
                    lamb=self.lamb,
                )
            elif self.device == 'cpu':
                (
                    self.cluster_centers_indices_,
                    self.labels_,
                    self.n_iter_,
                    self.A,
                    self.R
                ) = _affinity_propagation_cpu(
                    self.affinity_matrix_,
                    max_iter=self.max_iter,
                    convergence_iter=self.convergence_iter,
                    preference=preference,
                    damping=self.damping,
                    verbose=self.verbose,
                    return_n_iter=True,
                    random_state=random_state,
                    R_history=R_new,
                    alpha=self.alpha,
                    lamb=self.lamb,
                )
            else:
                raise NotImplementedError
            logging.info("Finish Incremental-AP Iteration.\n")
            
            tmp = self.A + self.R
            assert not np.isinf(tmp).any(), "Inf detected in Fixness\n"
            column_sum = tmp.sum(axis=0)
            row_sum = tmp.sum(axis=1)
            representation = (column_sum - row_sum + np.diag(tmp)) / tmp.shape[0]
            assert not np.isinf(representation).any(), "Inf detected in Representation\n"
            min_val = np.min(representation[:6000])
            max_val = np.max(representation)
            scaled_representation = (representation - min_val) / (max_val - min_val)
            scaled_representation[scaled_representation<0] = 0.
            score = scaled_representation
            if quality is not None:
                quality = self.quality_scores.detach().numpy().tolist() + quality
                quality = np.array(quality)
                min_val = np.min(quality)
                # min_val = np.mean(quality)
                # min_val = np.percentile(quality, 25)
                max_val = np.max(quality)
                scaled_quality = (quality - min_val) / (max_val - min_val)
                if self.mode == 'addition':
                    score = scaled_representation + self.gamma*scaled_quality
                elif self.mode == 'multiply':
                    score = (1+scaled_representation) * np.power(1+scaled_quality, self.gamma)
                elif 'nonlinear' in self.mode:
                    quantile = float(self.mode.split('nonlinear_')[-1])
                    ratio_upper = np.quantile(scaled_quality, quantile)
                    ratio_lower = np.quantile(scaled_quality, 0.3)

                    mul_coe = 4 / (ratio_upper - ratio_lower)
                    sub_coe = ratio_lower + 2 / mul_coe
                    scaled_quality = (scaled_quality - sub_coe) * mul_coe
                    scaled_quality =  1 / (1 + np.exp(-scaled_quality))

                    score = (1+scaled_representation) * (1+scaled_quality)
                else:
                    raise NotImplementedError
            _, self.cluster_centers_indices_ = torch.topk(torch.tensor(score), self.n_clusters)
            pool_indexes = []
            for idx in self.cluster_centers_indices_:
                if idx < self.n_clusters:
                    pool_indexes.append(self.pool_indexes[idx])
                else:
                    pool_indexes.append(self.last_count + idx - self.n_clusters)
            self.pool_indexes = np.array(pool_indexes).copy()
            del pool_indexes
            self.representative_scores = torch.tensor(scaled_representation)[self.cluster_centers_indices_].detach()
            self.overall_scores = torch.tensor(score)[self.cluster_centers_indices_].detach()
            if quality is not None:
                self.quality_scores = torch.tensor(quality)[self.cluster_centers_indices_].detach()

            tmp_save = {}
            tmp_save['representation'] = representation
            tmp_save['scaled_representation_scores'] = scaled_representation
            all_candidate_index = []
            for idx in range(representation.shape[0]):
                if idx < self.n_clusters:
                    all_candidate_index.append(self.pool_indexes[idx])
                else:
                    all_candidate_index.append(self.last_count + idx - self.n_clusters)
            tmp_save['all_candidate_index'] = all_candidate_index
            tmp_save['overall_scores'] = score
            if quality is not None:
                tmp_save['quality'] = quality
                tmp_save['scaled_quality_scores'] = torch.tensor(scaled_quality)
              
            if self.save_log:
                os.makedirs(f'./ap_logs/1223_cleaned_5sets_{self.affinity}_{self.mode}_alpha_{self.alpha}_lamb_{self.lamb}_gamma_{self.gamma}', exist_ok=True)
                torch.save(tmp_save, f'./ap_logs/1223_cleaned_5sets_{self.affinity}_{self.mode}_alpha_{self.alpha}_lamb_{self.lamb}_gamma_{self.gamma}/{self.iter_cnt}.pth')
            
            self.last_count += cur_num

        return self


    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, accept_sparse="csr")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError(
                "Predict method is not supported when affinity='precomputed'."
            )

        if self.cluster_centers_.shape[0] > 0:
            with config_context(assume_finite=True):
                return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn(
                (
                    "This model does not have any cluster centers "
                    "because affinity propagation did not converge. "
                    "Labeling every sample as '-1'."
                ),
                ConvergenceWarning,
            )
            return np.array([-1] * X.shape[0])

    def fit_predict(self, X, y=None):
        """Fit clustering from features/affinity matrix; return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
