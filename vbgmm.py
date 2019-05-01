import numpy as np
import copy
from scipy.special import digamma, logsumexp, gammaln
from sklearn.cluster import KMeans

class VariationalGMM():

    def __init__(self, n_components=3, max_iter=100, tolerance=1e-3, alpha_prior=1, beta_prior=1, dof=None,
                 wishart_matrix_prior=None, weights_prior=None, means_prior=None, covariances_prior=None):
        self.n_components = n_components  # Number of mixture components (K)
        self.max_iter = max_iter  # number of iterations to run iterative update of varational inference
        self.tolerance = tolerance  # Log-likelihood tolerance for terminating EM
        self.alpha_prior = alpha_prior  # Dirichlet parameter for prior of weights. (1, K)
        self.beta_prior = beta_prior  # scaling on precisions matrix.
        self.wishart_matrix_prior = wishart_matrix_prior
        self.dof = dof  # degrees of freedom for Wishart distribution
        self.covariances_prior = covariances_prior  # Initial covariances of mixands.
        self.weights_prior = weights_prior  # Initial weights of mixands
        self.means_prior = means_prior  # Initial means of mixands

    def fit(self, X):
        self._initialize_parameters(X)

        for n in range(0, self.max_iter):
            self.log_resp, self.resp = self.e_step(X)
            N_k, x_bar_k, S_k = self.m_step(X, self.resp)
            # check if convergence
            # ensure lowerbound increases
            print(self._calculate_lower_bound(N_k, x_bar_k, S_k))

    def _initialize_parameters(self, X):
        self.n_samples_ = np.shape(X)[0]  # number of samples
        self.n_features_ = np.shape(X)[1]  # number of features
        self.resp_ = np.zeros([self.n_samples_, self.n_components])
        self.alpha_k = np.full([self.n_components, ], self.alpha_prior)  # dirichlet parameters
        self.means = KMeans(self.n_components).fit(X).cluster_centers_.T # We default to initializing the means of our mixands with centers of KMeans model.
        self.means_prior = np.mean(X, axis=0)
        self.weights = self.weights_prior if self.weights_prior is not None else np.diag(np.random.uniform(0, 1, self.n_components))
        self.beta_k = np.full(self.n_components, self.beta_prior)  # scale of precision matrix.
        self.log_pi = digamma(self.alpha_k) - digamma(np.sum(self.alpha_k))
        self.log_lambda = np.zeros(self.n_components)
        self.dof = self.dof if self.dof is not None else self.n_features_ + 50
        self.nu_k = np.full([self.n_components, ], self.dof)
        self.W_k = np.zeros(
            [self.n_features_, self.n_features_, self.n_components])  # scaling matrix of wishart distribution
        self.W_prior = self.wishart_matrix_prior if self.wishart_matrix_prior is not None else np.diag(np.full(
            [self.n_features_, ], 100))
        self.W_prior_inv = np.linalg.inv(self.W_prior)  # Inverse of initial wishart component

        for k in range(0, self.n_components):
            self.W_k[:, :, k] = self.W_prior  # Scale matrix for Wishart
            self._update_expected_log_lambda()


    def e_step(self, X):
        # In the variational e_step, the ultimate goal is to calculate the responsibilities resp
        log_rho_nk = np.zeros([self.n_samples_, self.n_components])  # log rho, see Bishop 10.46
        for k in range(0, self.n_components):
            # Calculate the proportional responsiblities via 10.67 in Bishop.
            diff = X - self.means[:, k]
            log_rho_nk[:, k] = self.log_pi[k] + .5 * self.log_lambda[k] - .5 * (
                    self.n_features_ / self.beta_k[k]) - .5 * self.nu_k[k] * np.diag(
                np.dot(np.dot(diff, self.W_k[:, :, k]), np.transpose(diff)))
        Z = logsumexp(log_rho_nk, axis=1)
        print(log_rho_nk)
        log_resp = log_rho_nk - Z
        resp = np.exp(log_resp)
        return log_resp, resp

    def m_step(self, X, resp):

        N_k, x_bar_k, S_k = self._estimate_gaussian_mixture_parameters(X, resp)
        self._update_weights(N_k)
        self._update_expected_log_pi(N_k)
        self._update_means(N_k, x_bar_k)
        self._update_expected_log_lambda()
        self._update_gaussian_wishart(N_k, S_k, x_bar_k)
        return N_k, x_bar_k, S_k

    def _estimate_gaussian_mixture_parameters(self, X, resp):
        x_bar_k = np.zeros([self.n_features_, self.n_components])  # estimated centers of the component
        S_k = np.zeros(
            [self.n_features_, self.n_features_, self.n_components])  # estimated covariances of the components
        N_k = np.sum(resp,
                     0) + 1e-10  # sum or responsibilities for each component i.e. number of data samples in each component
        for k in range(0, self.n_components):
            x_bar_k[:, k] = np.dot(resp[:, k], X) / N_k[k]
            x_cen = X - x_bar_k[:, k]
            S_k[:, :, k] = np.dot(np.multiply(resp[:, k].reshape(272,1), x_cen).T, x_cen) / N_k[k]  # Bishop equation 10.53
        return N_k, x_bar_k, S_k

    def _update_weights(self, N_k):
        self.weights_ = (self.alpha_prior + N_k) / (self.n_components * self.alpha_prior + N_k)

    def _update_means(self, N_k, x_bar_k):
        for k in range(0, self.n_components):
            self.means[:, k] = (1 / self.beta_k[k]) * (
                    self.beta_prior * self.means_prior + N_k[k] * x_bar_k[:, k])  # from Bishop 10.61

    def _update_gaussian_wishart_parameters(self, N_k):
        self.beta_k = self.beta_prior + N_k  # from Bishop 10.60
        self.nu_k = self.dof + N_k + 1  # from Bishop 10.63, though according to sci-kit learn, it shouldn't have the +1?

    def _update_gaussian_wishart(self, N_k, S_k, x_bar_k):
        self._update_gaussian_wishart_parameters(N_k)
        for k in range(0, self.n_components):
            self.W_k[:, :, k] = self.W_prior_inv + N_k[k] * S_k[:, :, k] + (self.beta_prior * N_k[k]) \
                                / (self.beta_prior + N_k[k]) * np.dot(x_bar_k[:, k] \
                                                                      - self.means_prior, np.transpose(
                x_bar_k[:, k] - self.means_prior))  # from Bishop 10.62
            self.W_k[:, :, k] = np.linalg.inv(self.W_k[:, :, k])

    def _update_dirichlet_parameter(self, N_k):
        self.alpha_k = self.alpha_prior + N_k  # from Bishop 10.58

    def _update_expected_log_pi(self, N_k):
        self._update_dirichlet_parameter(N_k)
        self.log_pi = digamma(self.alpha_k) - digamma(np.sum(self.alpha_k))  # from Bishop 10.66

    def _update_expected_log_lambda(self, ):
        for k in range(0, self.n_components):
            digamma_sum = 0
            for i in range(1, self.n_features_ + 1):
                digamma_sum += digamma((self.nu_k[k] + 1 - i) / 2)
            self.log_lambda[k] = digamma_sum + self.n_features_ * np.log(2) + np.log(np.linalg.det(self.W_k[:, :, k]))

    def log_B(self, W, nu):
        n_col = np.shape(W)[0]

        gamma_sum = 0
        for i in range(0, n_col):
            gamma_sum += gammaln(0.5 * (nu + 1 - i))
        # Compute logB function via Bishop B.79
        return (-0.5 * nu * np.log(np.linalg.det(W)) - (0.5 * nu * n_col * np.log(2) + 0.25 * n_col * (n_col - 1) *
                                                        np.log(np.pi) + gamma_sum))  # log of B.79

    def _calculate_lower_bound(self, N_k, x_bar_k, S_k):
        log_px = 0
        log_pml = 0
        log_pml2 = 0
        log_qml = 0
        for k in range(0, self.n_components):
            # Here we collect all terms that require summations index by the k-th component.

            # see Bishop 10.71
            log_px = log_px + N_k[k] * (self.log_lambda[k] - self.n_features_ / self.beta_k[k] - self.nu_k[k] *
                                        np.trace(np.dot(S_k[:, :, k], self.W_k[:, :, k])) - self.nu_k[k] * np.dot(np.dot(np.transpose(
                        x_bar_k[:, k] - self.means[:, k]), self.W_k[:, :, k]),
                        x_bar_k[:, k] - self.means[:, k]) - self.n_features_ * np.log(2 * np.pi))

            # see Bishop 10.74
            log_pml = log_pml + self.n_features_ * np.log(self.beta_prior / (2 * np.pi)) + self.log_lambda[k] - \
            (self.n_features_ * self.beta_prior) / self.beta_k[k] - self.beta_prior * self.nu_k[k] * np.dot(
                np.dot(np.transpose(self.means[:, k] - self.means_prior),
                       self.W_k[:, :, k]), self.means[:, k] - self.means_prior)

            # see Bishop 10.74
            log_pml2 = log_pml2 + self.nu_k[k] * np.trace(np.dot(self.wishart_matrix_prior, self.W_k[:, :, k]))

            # see Bishop 10.77
            log_qml = log_qml + 0.5 * self.log_lambda[k] + 0.5 * self.n_features_ * np.log(self.beta_k[k] / (2 * np.pi)) \
                      - 0.5 * self.n_features_ - self.logB(W=self.W_k[:, :, k], nu=self.nu_k[k]) \
                      - 0.5 * (self.nu_k[k] - self.n_features_ - 1) * self.log_lambda[k] + 0.5 * self.nu_k[k] * self.n_features_

        log_px = 0.5 * log_px  # see Bishop 10.71
        log_pml = 0.5 * log_pml + self.n_components * self.logB(W=self.wishart_matrix_prior, nu=self.dof) + 0.5 * (
                    self.dof - self.n_features_ - 1) * np.sum(self.log_lambda) - 0.5 * log_pml2  # see Bishop 10.74
        log_pz = np.sum(np.transpose(self.resp, self.log_pi))  # see Bishop 10.72
        log_qz = np.sum(self.resp * self.log_resp)  # 10.75
        log_pp = np.sum((self.alpha_prior - 1) * self.log_pi) + gammaln(np.sum(self.n_components * self.alpha_prior)) - \
                 self.n_components * np.sum(gammaln(self.alpha_prior))  # 10.73

        log_qp = np.sum((self.alpha_k - 1) * self.log_pi) + gammaln(np.sum(self.alpha_k)) - np.sum(gammaln(self.alpha_k))  # 10.76


        # Sum all parts to compute lower bound
        return log_px + log_pz + log_pp + log_pml - log_qz - log_qp - log_qml
