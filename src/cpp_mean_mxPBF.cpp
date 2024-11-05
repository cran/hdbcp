#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include "mvrnorm.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// Function to compute a component of the Bayes Factor
double mean2_get_sigmasq(const arma::vec& x) {
  int n = x.n_elem;

  arma::vec vv(n, arma::fill::ones);

  double term1 = arma::dot(x, x);
  double term2 = std::pow(arma::dot(x, vv), 2.0) / arma::dot(vv, vv);

  double output = (term1 - term2) / static_cast<double>(n);
  return output;
}

// Function to compute the Bayes Factor for two samples within a window
// [[Rcpp::export]]
arma::vec cpp_mean2_mxPBF_single(const arma::mat& X, const arma::mat& Y, double log_gamma) {
  int n1 = X.n_rows; double nn1 = static_cast<double>(n1);
  int n2 = Y.n_rows; double nn2 = static_cast<double>(n2);
  int p  = X.n_cols;
  int n  = (n1 + n2); double nn  = static_cast<double>(n);

  arma::vec sigsqX(p, fill::zeros);
  arma::vec sigsqY(p, fill::zeros);
  arma::vec sigsqZ(p, fill::zeros);
  for (int i = 0; i < p; i++) {
    sigsqX(i) = mean2_get_sigmasq(X.col(i));
    sigsqY(i) = mean2_get_sigmasq(Y.col(i));
    sigsqZ(i) = mean2_get_sigmasq(join_vert(X.col(i), Y.col(i)));
  }

  arma::vec logBFvec(p, fill::zeros);
  double term1, term2;
  for (int i = 0; i < p; i++) {
    term1 = nn * sigsqZ(i);
    term2 = nn1 * sigsqX(i) + nn2 * sigsqY(i);
    logBFvec(i) = 0.5 * log_gamma + (nn / 2.0) * (std::log(term1 / term2));
  }
  return logBFvec;
}

// Function to compute Bayes factor for two samples within a window using a subset of column vectors
arma::vec cpp_mean2_mxPBF_approx(const arma::mat& X, const arma::mat& Y, double log_gamma, double quantile) {
  int p  = X.n_cols;

  arma::rowvec mean_X = arma::mean(X, 0);
  arma::rowvec mean_Y = arma::mean(Y, 0);
  arma::rowvec var_X = arma::var(X, 0, 0);
  arma::rowvec var_Y = arma::var(Y, 0, 0);

  arma::rowvec standardized_diff = arma::abs(mean_X / sqrt(var_X) - mean_Y / sqrt(var_Y));

  // standardized difference
  arma::uvec sorted_indices = arma::sort_index(standardized_diff, "descend");
  arma::uvec selected_columns = sorted_indices.head(std::ceil(p * quantile));

  int n1 = X.n_rows; double nn1 = static_cast<double>(n1);
  int n2 = Y.n_rows; double nn2 = static_cast<double>(n2);
  int n  = (n1 + n2); double nn  = static_cast<double>(n);
  int n_selected = selected_columns.n_elem;

  arma::vec sigsqX(n_selected, fill::zeros);
  arma::vec sigsqY(n_selected, fill::zeros);
  arma::vec sigsqZ(n_selected, fill::zeros);
  arma::vec logBFvec(n_selected, fill::zeros);
  double term1, term2;
  for (int i = 0; i < n_selected; ++i) {
    sigsqX(i) = mean2_get_sigmasq(X.col(selected_columns(i)));
    sigsqY(i) = mean2_get_sigmasq(Y.col(selected_columns(i)));
    sigsqZ(i) = mean2_get_sigmasq(join_vert(X.col(selected_columns(i)), Y.col(selected_columns(i))));

    term1 = nn * sigsqZ(i);
    term2 = nn1 * sigsqX(i) + nn2 * sigsqY(i);
    logBFvec(i) = 0.5 * log_gamma + (nn / 2.0) * (std::log(term1 / term2));
  }
  return logBFvec;
}

// Function to compute the mxPBF
// [[Rcpp::export]]
arma::vec cpd_mean_mxPBF(const arma::mat& X, int nw, double alp, int n_threads) {

  int n = X.n_rows;
  int p = X.n_cols;

  double gam = std::pow(std::max(2 * nw, p), -alp);
  double log_gam = std::log(gam / (gam + 1.0));

  arma::mat term2mat(n - 2 * nw + 1, p, fill::zeros);

#pragma omp parallel for num_threads(n_threads)
  for (int j = 0; j < p; j++) {
    arma::vec X_seg1 = X.col(j).subvec(0, 2 * nw - 1);
    double sum1 = arma::dot(X_seg1, X_seg1);

    arma::vec X_seg2 = X.col(j).subvec(0, nw - 1);
    double sum2 = arma::sum(X_seg2);

    arma::vec X_seg3 = X.col(j).subvec(nw, 2 * nw - 1);
    double sum3 = arma::sum(X_seg3);

    double sum4 = sum2 + sum3;

    double num = sum1 - (1.0 / (2 * nw)) * std::pow(sum4, 2);
    double denom = sum1 - (1.0 / nw) * std::pow(sum2, 2) - (1.0 / nw) * std::pow(sum3, 2);

    term2mat(0, j) = std::log(num / denom);

    for (int l = nw + 2; l <= (n - nw + 1); l++) {
      sum1 += std::pow(X(l + nw - 2, j), 2) - std::pow(X(l - nw - 2, j), 2);
      sum2 += X(l - 2, j) - X(l - nw - 2, j);
      sum3 += X(l + nw - 2, j) - X(l - 2, j);
      sum4 = sum2 + sum3;

      num = sum1 - (1.0 / (2 * nw)) * std::pow(sum4, 2);
      denom = sum1 - (1.0 / nw) * std::pow(sum2, 2) - (1.0 / nw) * std::pow(sum3, 2);
      term2mat(l - nw - 1, j) = std::log(num / denom);
    }
  }

  arma::vec term2maxvec = arma::max(term2mat, 1);

  arma::vec logmxPBF = 0.5 * log_gam + nw * term2maxvec;

  return(logmxPBF);
}

// Function to compute the mxPBF using simulated samples over a grid of alpha values
// [[Rcpp::export]]
arma::mat simulate_mxPBF_mean(const arma::mat& data, int nw, const arma::vec& alps, int n_samples, int n_threads) {
  int n = data.n_rows;
  int p = data.n_cols;
  int num_alps = alps.n_elem;
  double max_val = std::max(2 * nw, p);
  arma::vec gams = arma::exp(-alps * std::log(max_val));
  arma::vec log_gams = arma::log(gams / (gams + 1.0));
  arma::vec diffs = arma::diff(log_gams);
  arma::vec cumsum_diffs = arma::cumsum(diffs);
  arma::mat results(n_samples, num_alps, arma::fill::zeros);

  arma::rowvec mu = arma::mean(data, 0);
  arma::mat var = arma::cov(data);

  arma::vec eigval = arma::eig_sym(var);
  double min_eigval = eigval.min();

  if (min_eigval <= 1e-5) {
    var.diag() += std::abs(min_eigval) + 0.001;
  }

  for (int s = 0; s < n_samples; ++s) {
    arma::vec maxbf(num_alps, arma::fill::zeros);
    arma::mat sample = cpp_mvrnorm(n, mu.t(), var);
    arma::vec bf = cpd_mean_mxPBF(sample, nw, alps(0), n_threads);
    maxbf(0) = arma::max(bf);
    for (int k = 1; k < num_alps; ++k) {
      maxbf(k) = maxbf(0) + cumsum_diffs(k - 1) / 2;
    }
    results.row(s) = maxbf.t();
  }
  return results.t();
}
