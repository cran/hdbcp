#' Multivariate Normal Random Number Generator
#'
#' Generates random numbers from a multivariate normal distribution with specified mean and covariance matrix using a C++ implementation.
#'
#' @param n The number of random samples to generate. Defaults to 1.
#' @param mu The mean vector of the distribution.
#' @param Sigma The covariance matrix of the distribution.
#'
#' @return A numeric matrix where each row is a random sample from the multivariate normal distribution.
#'
#' @examples
#' # Example usage
#' mu <- c(0, 0)
#' Sigma <- matrix(c(1, 0.5, 0.5, 1), 2, 2)
#' mvrnorm_cpp(5, mu, Sigma)
#'
#' @export
mvrnorm_cpp <- function(n = 1, mu, Sigma) {
  return(cpp_mvrnorm(n, mu, Sigma))
}
