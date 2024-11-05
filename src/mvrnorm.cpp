#include <RcppArmadillo.h>
#include "mvrnorm.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// Function to generate multivariate normal sample using mvnrnd
// [[Rcpp::export]]
arma::mat cpp_mvrnorm(int n, const arma::vec& mu, const arma::mat& sigma) {
  return arma::mvnrnd(mu, sigma, n).t();
}
