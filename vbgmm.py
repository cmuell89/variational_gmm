import numpy as np
from scipy.special import digamma, logsumexp


class VariationalGMM():

    def __init__(self, n_components=1, max_iter=100, tolerance=1e-3, alpha_prior=1, beta_prior = 1, dof=None,
                 wishart_matrix_prior=None, weights_prior = None, means_prior = None, covariances_prior = None):
        self.n_components = n_components  # Number of mixture components (K)
        self.max_iter = max_iter # number of iterations to run iterative update of varational inference
        self.tolerance = tolerance  # Log-likelihood tolerance for terminating EM
        self.alpha_prior = alpha_prior # Dirichlet parameter for prior of weights. (1, K)
        self.beta_prior = beta_prior  # scaling on precisions matrix.
        self.wishart_matrix_prior = wishart_matrix_prior
        self.dof = dof  # degrees of freedom for Wishart distribution
        self.covariances_prior = covariances_prior  # Initial covariances of mixands.
        self.weights_prior = weights_prior  # Initial weights of mixands
        self.means_prior = means_prior  # Initial means of mixands


    def fit(self, X):
        self._initialize_parameters(X)

        for n in range(0, self.max_iter):
            log_resp, resp = self.e_step(X)
            self.m_step(X, resp)


    def _initialize_parameters(self, X):
        self.n_samples_ = np.shape(X)[0]  # number of samples
        self.n_features_ = np.shape(X)[1]  # number of features
        self.resp_ = np.zeros([self.n_samples_, self.n_components])
        self.alpha_k_ = np.full(self.n_components, self.alpha_prior)  # dirichlet parameters
        self.means = self.means_prior if self.means_prior is not None else np.mean(X, 0)
        self.weights = self.weights_prior if self.weights_prior is not None else
        self.beta_k = np.full(self.n_components, self.beta_prior)  # scale of precision matrix.
        self.log_pi = digamma(self.alpha_prior) - digamma(np.sum(self.alpha_prior))
        self.log_lambda = np.zeros(self.n_components)
        self.W_k = np.zeros([self.n_features_, self.n_features_, self.n_components])  # scaling matrix of wishart distribution
        self.W_prior =  self.wishart_matrix_prior if self.wishart_matrix_prior is not None else np.full(self.n_components, 100)
        self.W_prior_inv = np.inv(self.W_prior) # Inverse of initial wishart component

        log_pi = (digamma(self.alpha_prior) - digamma(np.sum(self.alpha_prior)))  # initial log_pi


    def e_step(self, X):
        # In the variational e_step, the ultimate goal is to calculate the responsibilities resp
        log_rho_nk = np.zeros([self.n_samples_, self.n_components])  # log rho, see Bishop 10.46
        for k in range(0, self.n_components):
            # Calculate the proportional responsiblities via 10.67 in Bishop.
            diff = X - self.means[k]
            log_rho_nk[:, k] = self.log_pi[k] + .5 * self.log_lambda[k] - .5 * (self.n_features_ / self.beta_k[k]) - .5 * self.nu_k[k] * np.diag(np.dot(np.dot(diff, self.W_k[:,:,k]), np.transpose(diff)))

        Z = logsumexp(log_rho_nk, axis=0)
        log_resp = log_rho_nk - Z
        resp = np.exp(log_resp)
        return log_resp, resp


    def m_step(self, X, resp):

        N_k, x_bar_k, S_k = self._estimate_gaussian_mixture_parameters(X, resp)
        # self._update_weights(N_k)
        self._update_expected_log_pi(N_k)
        self._update_means(N_k, x_bar_k)
        self._update_expected_log_lambda(N_k, x_bar_k, S_k)
        self._update_gaussian_wishart( N_k, S_k, x_bar_k)


    def _estimate_gaussian_mixture_parameters(self, X, resp):
        x_bar_k = np.zeros([self.n_features_, self.n_components]) # estimated centers of the component
        S_k = np.zeros([self.n_features_, self.n_features_, self.n_components]) # estimated covariances of the components
        N_k = np.sum(resp, 0) + 1e-10 # sum or responsibilities for each component i.e. number of data samples in each component
        for k in range(0, self.n_components):
            x_bar_k[:, k] = np.dot(resp[:, k], X) / N_k[k]
            x_cen = X - x_bar_k[:, k]
            S_k[:, :, k] = np.dot(np.transpose(x_cen), np.dot(x_cen, resp[:, k]))  # Bishop equation 10.53
        return N_k, x_bar_k, S_k


    def _update_dirichlet_parameter(self, N_k):
        self.alpha_k = self.alpha_prior + N_k  # from Bishop 10.58

    # def _update_weights(self, N_k):
    #     self.weights_ = (self.alpha_prior + N_k) / (self.n_components * self.alpha_prior + N_k)

    def _update_means(self, N_k, x_bar_k):
        for k in range(0, self.n_components):
            self.means[:, k] = (1 / self.beta_k[k]) * (self.beta_prior * self.means_prior + N_k[k] * x_bar_k[:, k])  # from Bishop 10.61

    def _update_gaussian_wishart_parameters(self, N_k):
        self.beta_k = self.beta_prior + N_k # from Bishop 10.60
        self.nu_k = self.dof + N_k + 1 # from Bishop 10.63, though according to sci-kit learn, it shouldn't have the +1?

    def _update_gaussian_wishart(self, N_k, S_k, x_bar_k):
        self._update_gaussian_wishart_parameters(N_k)
        for  k in range(0, self.n_components):
            self.W_k[:, :, k] = self.W_prior_inv + N_k[k] * S_k[:, :, k] + (self.beta_prior * N_k[k]) \
                                / (self.beta_prior + N_k[k]) * np.dot(x_bar_k[:, k] \
                                - self.means_prior, np.transpose(x_bar_k[:, k] - self.means_prior)) # from Bishop 10.62
            self.W_k[:, :, k] = np.linalg.inv(self.W_k[:, :, k])

    def _update_expected_log_pi(self, N_k):
        self._update_dirichlet_parameter(N_k)
        self.log_pi = digamma(self.alpha_k) - digamma(np.sum(self.alpha_k)) # from Bishop 10.66

    def _update_expected_log_lambda(self, N_k, x_bar_k, S_k):
        for k in range(0, self.n_components):
            digamma_sum = 0
            for i in range(0, self.n_features_):
                digamma_sum += digamma((self.nu_k[k] + 1 - i)/2)
            self.log_lambda[k] = digamma_sum + self.n_features_ * np.log(2) + np.log(np.linalg.det(self.W_k[:, :, k]))

    def _calculate_lower_bound(self, N_k, x_bar_k):
        log_px = 0
        log_pml = 0
        lb_qml = 0
        for k in range(0, self.n_components):
            # 10.71
            log_px = log_px + N_k[k] * (self.log_lambda[k] - self.n_features_ / self.beta_k[k] - self.nu_k[k] *
                                        np.trace(np.dot(S_k[:, :, k],  self.W_k[:, :, k])) - self.nu_k[k] * np.transpose(x_bar_k[:, k] -
            np.dot(np.dot(self.means[:, k]), self.W_k[:, :, k]), x_bar_k[:, k] - self.means[:, k]) - self.n_features_ * np.log(2 * np.pi) )

