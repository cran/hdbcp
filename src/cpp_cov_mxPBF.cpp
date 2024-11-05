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
double cov2_get_tau(const arma::vec& xi, const arma::vec& xj) {
  int n = xi.n_elem; double nn = static_cast<double>(n);

  double dotij = arma::dot(xi, xj);
  double term1 = arma::dot(xi, xi);
  double term2 = (dotij * dotij) / arma::dot(xj, xj);

  double output = (term1 - term2) / nn;
  return output;
}

// Function to compute the Bayes Factor for two samples within a window
// [[Rcpp::export]]
arma::mat cpp_cov2_mxPBF_single(const arma::mat& X, const arma::mat& Y, double a0, double b0, double log_gamma) {
  int n1 = X.n_rows; double nn1 = static_cast<double>(n1);
  int n2 = Y.n_rows; double nn2 = static_cast<double>(n2);
  int p = X.n_cols;
  int n = n1 + n2; double nn = static_cast<double>(n);

  double term_const = 0.5 * log_gamma + std::lgamma((nn1 / 2.0) + a0) + std::lgamma((nn2 / 2.0) + a0) - std::lgamma(nn / 2.0 + a0) + a0 * std::log(b0) - std::lgamma(a0);

  arma::mat logBFmat(p, p);
  logBFmat.fill(-10000);

  for (int i = 0; i < p; i++) {
    arma::vec Xi = X.col(i);
    arma::vec Yi = Y.col(i);
    arma::vec Zi = arma::join_vert(Xi, Yi);

    for (int j = 0; j < p; j++) {
      if (i != j) {
        arma::vec Xj = X.col(j);
        arma::vec Yj = Y.col(j);
        arma::vec Zj = arma::join_vert(Xj, Yj);

        double term1 = (nn1 / 2.0 + a0) * std::log(b0 + (nn1 / 2.0) * cov2_get_tau(Xi, Xj));
        double term2 = (nn2 / 2.0 + a0) * std::log(b0 + (nn2 / 2.0) * cov2_get_tau(Yi, Yj));
        double term3 = (nn / 2.0 + a0) * std::log(b0 + (nn / 2.0) * cov2_get_tau(Zi, Zj));

        logBFmat(i, j) = term_const - (term1 + term2) + term3;
      }
    }
  }

  return logBFmat;
}

// Function to compute mxPBF
// [[Rcpp::export]]
arma::vec cpd_cov_mxPBF(const arma::mat& X, double a0, double b0, int nw, double alp, int n_threads) {

  int n = X.n_rows;
  int p = X.n_cols;

  double gam = std::pow(std::max(2 * nw, p), -alp);
  double log_gam = std::log(gam / (gam + 1.0));
  double term_const = 0.5 * log_gam + 2.0 * std::lgamma(0.5 * nw + a0) - std::lgamma(nw + a0) - std::lgamma(a0) + a0 * std::log(b0);

  arma::cube valuecube(p, p, n - 2 * nw + 1, arma::fill::zeros);
  arma::cube term1cube(p, p, n - 2 * nw + 1, fill::zeros);

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

// Function to compute an approximate mxPBF, allowing for a small margin of error
arma::vec cpd_cov_mxPBF_approx(const arma::mat& X, double a0, double b0, int nw, double alp, int n_threads) {

  int n = X.n_rows;
  int p = X.n_cols;

  double gam = std::pow(std::max(2 * nw, p), -alp);
  double log_gam = std::log(gam / (gam + 1.0));
  double term_const = 0.5 * log_gam + 2.0 * std::lgamma(0.5 * nw + a0) - std::lgamma(nw + a0) - std::lgamma(a0) + a0 * std::log(b0);

  arma::cube valuecube(p, p, n - 2 * nw + 1, arma::fill::zeros);

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
      valuecube(j, i, 0) = std::pow(term4, 2) / (term5 * term6);

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
  }

  arma::vec logmxPBF = term_const + (0.5 * nw + a0) * arma::log(maxvec);

  return logmxPBF;
}

// Function to compute the mxPBF using simulated samples over a grid of alpha values
// [[Rcpp::export]]
arma::mat simulate_mxPBF_cov(const arma::mat& data, double a0, double b0, int nw, const arma::vec& alps, int n_samples, int n_threads) {
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
    arma::vec bf = cpd_cov_mxPBF(sample, a0, b0, nw, alps(0), n_threads);
    maxbf(0) = arma::max(bf);
    for (int k = 1; k < num_alps; ++k) {
      maxbf(k) = maxbf(0) + cumsum_diffs(k - 1) / 2;
    }
    results.row(s) = maxbf.t();
  }

  return results.t();
}

// An approximate version of the above
arma::mat simulate_mxPBF_cov_approx(const arma::mat& data, double a0, double b0, int nw, const arma::vec& alps, int n_samples, int n_threads) {
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
    arma::vec bf = cpd_cov_mxPBF_approx(sample, a0, b0, nw, alps(0), n_threads);
    maxbf(0) = arma::max(bf);
    for (int k = 1; k < num_alps; ++k) {
      maxbf(k) = maxbf(0) + cumsum_diffs(k - 1) / 2;
    }
    results.row(s) = maxbf.t();
  }

  return results.t();
}
