import numpy
from scipy.special import logsumexp
from sklearn.mixture.tests.test_gaussian_mixture import RandomData


class VariationalGMM():

    def __init__(self, n_components, tolerance, prior_weights, prior_means, prior_covariances):
        # Number of mixture components
        n_components = n_components
        # Log-liklihood tolerance for terminating EM
        tolerance = tolerance
        prior_covariances = prior_covariances
        prior_weights = prior_weights
        prior_means = prior_means

    def _estimate_parameters(self, X, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array.
        resp : array-like, shape (n_samples, n_components)
            The responsibilities for each data sample in X.
        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        nk : array-like, shape (n_components,)
            The numbers of data samples in the current components.
        means : array-like, shape (n_components, n_features)
            The centers of the current components.
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = self._estimate_covariances(resp, X, nk, means, reg_covar)
        return nk, means, covariances

    def _estimate_covariances(self, resp, X, nk, means, reg_covar):
        """
        Estimate the full covariance matrices.
        Parameters
        ----------
        resp : array-like, shape (n_samples, n_components)
        X : array-like, shape (n_samples, n_features)
        nk : array-like, shape (n_components,)
        means : array-like, shape (n_components, n_features)
        reg_covar : float
        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[::n_features + 1] += reg_covar
        return covariances

    def _estimate_weights(self, nk):
         """Estimate the parameters of the Dirichlet distribution.
         Parameters
         ----------
         nk : array-like, shape (n_components,)
         """
         # case Variationnal Gaussian mixture with dirichlet distribution
         self.weight_concentration_ = self.weight_concentration_prior_ + nk

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

     def _estimate_weighted_log_prob(self, X):
         """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
         Parameters
         ----------
         X : array-like, shape (n_samples, n_features)
         Returns
         -------
         weighted_log_prob : array, shape (n_samples, n_component)
         """
         return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == 'dirichlet_process':
            digamma_sum = digamma(self.weight_concentration_[0] +
                                  self.weight_concentration_[1])
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (digamma_a - digamma_sum +
                    np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            return (digamma(self.weight_concentration_) -
                    digamma(np.sum(self.weight_concentration_)))

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (self.degrees_of_freedom_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / self.mean_precision_)

    def _estimate_log_gaussian_prob(self, X, means, precisions_chol):
        """Estimate the log Gaussian probability.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        means : array-like, shape (n_components, n_features)
        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            shape of (n_components, n_features, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

        return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det_chol

    def _compute_precision_cholesky(self, covariances):
        """Compute the Cholesky decomposition of the precisions.
        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            cov_chol = linalg.cholesky(covariance, lower=True)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
        return precisions_chol

    def _e_step(self, X):
        """E step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

