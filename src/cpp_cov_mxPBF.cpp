#include <RcppArmadillo.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mvrnorm.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// Function to pre-compute cumulative sums
// [[Rcpp::export]]
void pre_compute_sums(const arma::mat& X, arma::mat& S_ii, arma::mat& S_i, arma::cube& S_ij) {
  int n = X.n_rows;
  int p = X.n_cols;

  S_ii.rows(1, n) = arma::cumsum(arma::square(X), 0);
  S_i.rows(1, n) = arma::cumsum(X, 0);

  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j) {
      arma::vec extended_slice(n + 1, arma::fill::zeros);
      extended_slice.subvec(1, n) = arma::cumsum(X.col(i) % X.col(j));
      S_ij.tube(i, j) = extended_slice;
    }
  }
}

// Function to Compute mxPBF for given data matrix
// [[Rcpp::export]]
arma::vec cpd_cov_mxPBF(const arma::mat& X, arma::mat& S_ii, arma::mat& S_i, arma::cube& S_ij, double a0, double b0, int nw, double alp, int n_threads) {
  int n = X.n_rows;
  int p = X.n_cols;

  double gam = std::pow(std::max(2 * nw, p), -alp);
  double log_gam = std::log(gam / (gam + 1.0));
  double const_term = 0.5 * log_gam + 2.0 * std::lgamma(0.5 * nw + a0) - std::lgamma(nw + a0) - std::lgamma(a0) + a0 * std::log(b0);

  arma::vec mxPBFs(n - 2 * nw + 1, arma::fill::zeros);

#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
  for (int l = nw + 1; l <= n - nw + 1; ++l) {
    double mxPBF = -arma::datum::inf;
    for (int i = 0; i < p; ++i) {
      for (int j = i + 1; j < p; ++j){
        // Compute combined window sums
        double sum_ii_c = S_ii(l + nw - 1, i) - S_ii(l - nw - 1, i);
        double sum_jj_c = S_ii(l + nw - 1, j) - S_ii(l - nw - 1, j);
        double sum_ij_c = S_ij(i, j, l + nw - 1) - S_ij(i, j, l - nw - 1);

        double term_i_c = b0 + 0.5 * (sum_ii_c - sum_ij_c * sum_ij_c / sum_jj_c);
        double term_j_c = b0 + 0.5 * (sum_jj_c - sum_ij_c * sum_ij_c / sum_ii_c);

        // Compute left window sums
        double sum_ii_l = S_ii(l - 1, i) - S_ii(l - nw - 1, i);
        double sum_jj_l = S_ii(l - 1, j) - S_ii(l - nw - 1, j);
        double sum_ij_l = S_ij(i, j, l - 1) - S_ij(i, j, l - nw - 1);

        double term_i_l = b0 + 0.5 * (sum_ii_l - sum_ij_l * sum_ij_l / sum_jj_l);
        double term_j_l = b0 + 0.5 * (sum_jj_l - sum_ij_l * sum_ij_l / sum_ii_l);

        // Compute right window sums
        double sum_ii_r = S_ii(l + nw - 1, i) - S_ii(l - 1, i);
        double sum_jj_r = S_ii(l + nw - 1, j) - S_ii(l - 1, j);
        double sum_ij_r = S_ij(i, j, l + nw - 1) - S_ij(i, j, l - 1);

        double term_i_r = b0 + 0.5 * (sum_ii_r - sum_ij_r * sum_ij_r / sum_jj_r);
        double term_j_r = b0 + 0.5 * (sum_jj_r - sum_ij_r * sum_ij_r / sum_ii_r);

        // Compute PBF
        double PBF_i = (nw + a0) * log(term_i_c) - (nw / 2.0 + a0) * log(term_i_l * term_i_r);
        double PBF_j = (nw + a0) * log(term_j_c) - (nw / 2.0 + a0) * log(term_j_l * term_j_r);

        // Update the maximum PBF for this l
        mxPBF = std::max({mxPBF, PBF_i, PBF_j});
      }
    }
    mxPBFs(l - nw - 1) = const_term + mxPBF;
  }
  return mxPBFs;
}

// Function to Implement Empirical FPR method
// [[Rcpp::export]]
arma::mat simulate_mxPBF_cov(const arma::mat& X, arma::mat& S_ii, arma::mat& S_i, arma::cube& S_ij, double a0, double b0, int nw, const arma::vec& alps, int n_samples, int n_threads) {
  int n = X.n_rows;
  int p = X.n_cols;

  int num_alps = alps.n_elem;
  double max_val = std::max(2 * nw, p);
  arma::vec gams = arma::exp(-alps * std::log(max_val));
  arma::vec log_gams = arma::log(gams / (gams + 1.0));
  arma::vec diffs = arma::diff(log_gams);
  arma::vec cumsum_diffs = arma::cumsum(diffs);
  arma::mat results(n_samples, num_alps, arma::fill::zeros);

  // Calculate sample mean and variance
  arma::rowvec mu = arma::mean(X, 0);
  arma::mat var = arma::cov(X);

  // Check if the covariance matrix is positive definite
  arma::vec eigval = arma::eig_sym(var);
  double min_eigval = eigval.min();

  if (min_eigval <= 1e-5) {
    // Adjust the diagonal elements
    var.diag() += std::abs(min_eigval) + 0.001;
  }

  // Generate samples and compute mxPBFs
  for (int s = 0; s < n_samples; ++s) {
    arma::vec mxPBF_alpha_vals(num_alps, arma::fill::zeros);
    arma::mat sample = cpp_mvrnorm(n, mu.t(), var);
    pre_compute_sums(sample, S_ii, S_i, S_ij);
    arma::vec mxPBF_alpha_1 = cpd_cov_mxPBF(sample, S_ii, S_i, S_ij, a0, b0, nw, alps(0), n_threads);
    mxPBF_alpha_vals(0) = arma::max(mxPBF_alpha_1);
    for (int k = 1; k < num_alps; ++k) {
      mxPBF_alpha_vals(k) = mxPBF_alpha_vals(0) + cumsum_diffs(k - 1) / 2;
    }
    results.row(s) = mxPBF_alpha_vals.t();
  }

  return results.t();
}

// Function to Compute mxPBF for given data matrix with centering applied (Not used)
arma::vec cpd_cov_mxPBF_centered(const arma::mat& X, arma::mat& S_ii, arma::mat& S_i, arma::cube& S_ij, double a0, double b0, int nw, int maxnw, double alp, int n_threads) {
  int n = X.n_rows;
  int p = X.n_cols;

  double gam = std::pow(std::max(2 * nw, p), -alp);
  double log_gam = std::log(gam / (gam + 1.0));
  double const_term = 0.5 * log_gam + 2.0 * std::lgamma(0.5 * nw + a0) - std::lgamma(nw + a0) - std::lgamma(a0) + a0 * std::log(b0);

  arma::vec mxPBFs(n - 2 * nw + 1, arma::fill::zeros);

#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
  for (int l = maxnw + 1; l <= n - maxnw + 1; ++l) {
    double mxPBF = -arma::datum::inf;
    for (int i = 0; i < p; ++i) {
      double sum_i_r = S_i(l + maxnw - 1, i) - S_i(l - 1, i);
      double mean_ii_r = sum_i_r * sum_i_r / (maxnw);
      double sum_i_l = S_i(l - 1, i) - S_i(l - maxnw - 1, i);
      double mean_ii_l = sum_i_l * sum_i_l / (maxnw);
      for (int j = i + 1; j < p; ++j){
        // Compute the mean over each window
        double sum_j_r = S_i(l + maxnw - 1, j) - S_i(l - 1, j);
        double mean_jj_r = sum_j_r * sum_j_r / (maxnw);
        double mean_ij_r = sum_i_r * sum_j_r / (maxnw);
        double sum_j_l = S_i(l - 1, j) - S_i(l - maxnw - 1, j);
        double mean_jj_l = sum_j_l * sum_j_l / (maxnw);
        double mean_ij_l = sum_i_r * sum_j_r / (maxnw);

        // Compute left window sums
        double sum_ii_l = S_ii(l - 1, i) - S_ii(l - nw - 1, i) - mean_ii_l;
        double sum_jj_l = S_ii(l - 1, j) - S_ii(l - nw - 1, j) - mean_jj_l;
        double sum_ij_l = S_ij(i, j, l - 1) - S_ij(i, j, l - nw - 1) - mean_ij_l;

        double term_i_l = b0 + 0.5 * (sum_ii_l - sum_ij_l * sum_ij_l / sum_jj_l);
        double term_j_l = b0 + 0.5 * (sum_jj_l - sum_ij_l * sum_ij_l / sum_ii_l);

        // Compute right window sums
        double sum_ii_r = S_ii(l + nw - 1, i) - S_ii(l - 1, i) - mean_ii_r;
        double sum_jj_r = S_ii(l + nw - 1, j) - S_ii(l - 1, j) - mean_jj_r;
        double sum_ij_r = S_ij(i, j, l + nw - 1) - S_ij(i, j, l - 1) - mean_ij_r;

        double term_i_r = b0 + 0.5 * (sum_ii_r - sum_ij_r * sum_ij_r / sum_jj_r);
        double term_j_r = b0 + 0.5 * (sum_jj_r - sum_ij_r * sum_ij_r / sum_ii_r);

        // Compute combined window sums
        double sum_ii_c = sum_ii_l + sum_ii_r;
        double sum_jj_c = sum_jj_l + sum_jj_r;
        double sum_ij_c = sum_ij_l + sum_ij_r;

        double term_i_c = b0 + 0.5 * (sum_ii_c - sum_ij_c * sum_ij_c / sum_jj_c);
        double term_j_c = b0 + 0.5 * (sum_jj_c - sum_ij_c * sum_ij_c / sum_ii_c);

        // Compute PBF
        double PBF_i = (nw + a0) * log(term_i_c) - (nw / 2.0 + a0) * log(term_i_l * term_i_r);
        double PBF_j = (nw + a0) * log(term_j_c) - (nw / 2.0 + a0) * log(term_j_l * term_j_r);

        // Update the maximum PBF for this l
        mxPBF = std::max({mxPBF, PBF_i, PBF_j});
      }
    }
    mxPBFs(l - nw - 1) = const_term + mxPBF;
  }
  return mxPBFs;
}


// Function to Implement Empirical FPR method with centering applied (Not used)
arma::mat simulate_mxPBF_cov_centered(const arma::mat& X, arma::mat& S_ii, arma::mat& S_i, arma::cube& S_ij, double a0, double b0, int nw, int maxnw, const arma::vec& alps, int n_samples, int n_threads) {
  int n = X.n_rows;
  int p = X.n_cols;

  int num_alps = alps.n_elem;
  double max_val = std::max(2 * nw, p);
  arma::vec gams = arma::exp(-alps * std::log(max_val));
  arma::vec log_gams = arma::log(gams / (gams + 1.0));
  arma::vec diffs = arma::diff(log_gams);
  arma::vec cumsum_diffs = arma::cumsum(diffs);
  arma::mat results(n_samples, num_alps, arma::fill::zeros);

  // Calculate sample mean and variance
  arma::rowvec mu = arma::mean(X, 0);
  arma::mat var = arma::cov(X);

  // Check if the covariance matrix is positive definite
  arma::vec eigval = arma::eig_sym(var);
  double min_eigval = eigval.min();

  if (min_eigval <= 1e-5) {
    // Adjust the diagonal elements
    var.diag() += std::abs(min_eigval) + 0.001;
  }

  // Generate samples and compute mxPBFs
  for (int s = 0; s < n_samples; ++s) {
    arma::vec mxPBF_alpha_vals(num_alps, arma::fill::zeros);
    arma::mat sample = cpp_mvrnorm(n, mu.t(), var);
    pre_compute_sums(sample, S_ii, S_i, S_ij);
    arma::vec mxPBF_alpha_1 = cpd_cov_mxPBF_centered(sample, S_ii, S_i, S_ij, a0, b0, nw, maxnw, alps(0), n_threads);
    mxPBF_alpha_vals(0) = arma::max(mxPBF_alpha_1);
    for (int k = 1; k < num_alps; ++k) {
      mxPBF_alpha_vals(k) = mxPBF_alpha_vals(0) + cumsum_diffs(k - 1) / 2;
    }
    results.row(s) = mxPBF_alpha_vals.t();
  }

  return results.t();
}

// Function to Compute mxPBF for given data matrix (Not used)
arma::vec cpd_cov_mxPBF2_previous(const arma::mat& X, double a0, double b0, int nw, double alp, int n_threads) {

  int n = X.n_rows;
  int p = X.n_cols;

  double gam = std::pow(std::max(2 * nw, p), -alp);
  double log_gam = std::log(gam / (gam + 1.0));
  double term_const = 0.5 * log_gam + 2.0 * std::lgamma(0.5 * nw + a0) - std::lgamma(nw + a0) - std::lgamma(a0) + a0 * std::log(b0);

  arma::cube valuecube(p, p, n - 2 * nw + 1, arma::fill::zeros);
  arma::cube term1cube(p, p, n - 2 * nw + 1, arma::fill::zeros);

#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
  for (int i = 0; i < p; i++) {
    for (int j = i + 1; j < p; j++) {
      double sum1_i2 = arma::sum(arma::square(X.rows(0, 2 * nw - 1).col(i)));
      double sum2_i2 = arma::sum(arma::square(X.rows(0, nw - 1).col(i)));
      double sum3_i2 = arma::sum(arma::square(X.rows(nw, 2 * nw - 1).col(i)));

      double sum1_j2 = arma::sum(arma::square(X.rows(0, 2 * nw - 1).col(j)));
      double sum2_j2 = arma::sum(arma::square(X.rows(0, nw - 1).col(j)));
      double sum3_j2 = arma::sum(arma::square(X.rows(nw, 2 * nw - 1).col(j)));

      double sum2_ij = arma::dot(X.rows(0, nw - 1).col(i), X.rows(0, nw - 1).col(j));
      double sum3_ij = arma::dot(X.rows(nw, 2 * nw - 1).col(i), X.rows(nw, 2 * nw - 1).col(j));
      double sum1_ij = sum2_ij + sum3_ij;

      double term1 = b0 + 0.5 * (sum1_i2 - std::pow(sum1_ij, 2) / sum1_j2);
      double term2 = b0 + 0.5 * (sum2_i2 - std::pow(sum2_ij, 2) / sum2_j2);
      double term3 = b0 + 0.5 * (sum3_i2 - std::pow(sum3_ij, 2) / sum3_j2);

      double term4 = b0 + 0.5 * (sum1_j2 - std::pow(sum1_ij, 2) / sum1_i2);
      double term5 = b0 + 0.5 * (sum2_j2 - std::pow(sum2_ij, 2) / sum2_i2);
      double term6 = b0 + 0.5 * (sum3_j2 - std::pow(sum3_ij, 2) / sum3_i2);

      valuecube(i, i, 0) = -10000;
      valuecube(i, j, 0) = std::pow(term1, 2) / (term2 * term3);
      term1cube(i, j, 0) = term1;
      valuecube(j, i, 0) = std::pow(term4, 2) / (term5 * term6);
      term1cube(j, i, 0) = term4;

      for (int l = nw + 2; l <= (n - nw + 1); l++) {
        double new_i2 = std::pow(X(l + nw - 2, i), 2);
        double old_i2 = std::pow(X(l - nw - 2, i), 2);
        double new_j2 = std::pow(X(l + nw - 2, j), 2);
        double old_j2 = std::pow(X(l - nw - 2, j), 2);

        sum1_i2 += new_i2 - old_i2;
        sum2_i2 += std::pow(X(l - 2, i), 2) - old_i2;
        sum3_i2 += new_i2 - std::pow(X(l - 2, i), 2);

        sum1_j2 += new_j2 - old_j2;
        sum2_j2 += std::pow(X(l - 2, j), 2) - old_j2;
        sum3_j2 += new_j2 - std::pow(X(l - 2, j), 2);

        sum2_ij += X(l - 2, i) * X(l - 2, j) - X(l - nw - 2, i) * X(l - nw - 2, j);
        sum3_ij += X(l + nw - 2, i) * X(l + nw - 2, j) - X(l - 2, i) * X(l - 2, j);
        sum1_ij = sum2_ij + sum3_ij;

        term1 = b0 + 0.5 * (sum1_i2 - std::pow(sum1_ij, 2) / sum1_j2);
        term2 = b0 + 0.5 * (sum2_i2 - std::pow(sum2_ij, 2) / sum2_j2);
        term3 = b0 + 0.5 * (sum3_i2 - std::pow(sum3_ij, 2) / sum3_j2);

        term4 = b0 + 0.5 * (sum1_j2 - std::pow(sum1_ij, 2) / sum1_i2);
        term5 = b0 + 0.5 * (sum2_j2 - std::pow(sum2_ij, 2) / sum2_i2);
        term6 = b0 + 0.5 * (sum3_j2 - std::pow(sum3_ij, 2) / sum3_i2);

        valuecube(i, i, l - nw - 1) = -10000;
        valuecube(i, j, l - nw - 1) = std::pow(term1, 2) / (term2 * term3);
        valuecube(j, i, l - nw - 1) = std::pow(term4, 2) / (term5 * term6);
        term1cube(i, j, l - nw - 1) = term1;
        term1cube(j, i, l - nw - 1) = term4;
      }
    }
  }

  int num_slices = valuecube.n_slices;
  arma::vec maxvec(num_slices, arma::fill::zeros);
  arma::vec maxterm1vec(num_slices, arma::fill::zeros);
  for (int k = 0; k < num_slices; ++k) {
    arma::mat current_slice = valuecube.slice(k);
    arma::uword max_index = current_slice.index_max();
    maxvec(k) = current_slice(max_index);
    maxterm1vec(k) = term1cube.slice(k)(max_index);
  }

  arma::vec logmxPBF = term_const + (0.5 * nw) * arma::log(maxvec) + a0 * arma::log(maxvec / maxterm1vec);

  return logmxPBF;
}
