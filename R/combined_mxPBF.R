#' Change Point Detection in Mean Structure using Maximum Pairwise Bayes Factor (mxPBF)
#'
#' This function detects change points in both mean and covariance structure of multivariate Gaussian data using the Maximum Pairwise Bayes Factor (mxPBF).
#' The function selects alpha that controls the empirical False Positive Rate (FPR), as suggested in the paper.
#' The function conducts a multiscale approach using the function.
#'
#' @param given_data An \eqn{(n \times p)} data matrix representing \eqn{n} observations and \eqn{p} variables.
#' @param a0 A hyperparameter \eqn{a_0} used in the mxPBF (default: 0.01).
#' @param b0 A hyperparameter \eqn{b_0} used in the mxPBF (default: 0.01).
#' @param nws A set of window sizes for change point detection.
#' @param alps A grid of alpha values used in the empirical False Positive Rate (FPR) method.
#' @param FPR_want Desired False Positive Rate for selecting alpha, used in the empirical FPR method (default: 0.05).
#' @param n_sample Number of simulated samples to estimate the empirical FPR, used in the empirical FPR method (default: 300).
#' @param n_cores Number of threads for parallel execution via OpenMP (default: 1).
#' @param centering Method for centering the data if it has a nonzero mean before analysis. Can be one of \code{"mean"}, \code{"median"}, or \code{"skip"} (default: "skip").
#'
#' @return A list provided. Each element in the list contains: \describe{
#' \item{Result_cov}{A list result from the \code{mxPBF_cov()} function.}
#' \item{Result_mean}{A list result from the \code{mxPBF_mean()} function applied to each segmented data.}
#' \item{Change_points_cov}{Locations of detected change points identified by \code{mxPBF_cov()} function.}
#' \item{Change_points_mean}{Locations of detected change points identified by \code{mxPBF_mean()} function.}
#' }
#'
#' @examples
#' \donttest{
#' nws <- c(25, 60, 100)
#' alps <- seq(1,10,0.05)
#' ## H0 data
#' mu1 <- rep(0,10)
#' sigma1 <- diag(10)
#' X <- mvrnorm_cpp(500, mu1, sigma1)
#' res1 <- mxPBF_combined(X, nws = nws, alps = alps)
#'
#' ## H1 data
#' mu2 <- rep(1,10)
#' sigma2 <- diag(10)
#' for (i in 1:10) {
#'   for (j in i:10) {
#'     if (i == j) {
#'     next
#'     } else {
#'     cov_value <- rnorm(1, 1, 1)
#'     sigma2[i, j] <- cov_value
#'     sigma2[j, i] <- cov_value
#'     }
#'   }
#' }
#' sigma2 <- sigma2 + (abs(min(eigen(sigma2)$value))+0.1)*diag(10) # Make it nonsingular
#' Y1 <- mvrnorm_cpp(150, mu1, sigma1)
#' Y2 <- mvrnorm_cpp(150, mu2, sigma1)
#' Y3 <- mvrnorm_cpp(200, mu2, sigma2)
#' Y <- rbind(Y1, Y2, Y3)
#' res2 <- mxPBF_combined(Y, nws = nws, alps = alps)
#' }
#'
#' @export
mxPBF_combined <- function(given_data, a0 = 0.01, b0 = 0.01, nws, alps, FPR_want = 0.05, n_sample = 300, n_cores = 1, centering = "skip"){
  n <- nrow(given_data)
  # Centering
  if (centering == "mean") {
    means <- colMeans(given_data)
    centered_data <- sweep(given_data, 2, means, FUN = "-")
  }
  if (centering == "median") {
    medians <- apply(given_data, 2, median)
    centered_data <- sweep(given_data, 2, medians, FUN = "-")
  }
  if (centering == "skip") {
    centered_data <- given_data
  }
  # Applying covariance method
  res_cov <- mxPBF_cov(given_data = centered_data, a0 = a0, b0 = b0, nws = nws, alps = alps, FPR_want = FPR_want, n_sample = n_sample, n_cores = n_cores, centering = "skip")

  changes_cov <- majority_rule_mxPBF(res_cov, nws, n)

  res_mean_list <- list()
  changes_mean <- list()
  segment_points <- c(1, sort(changes_cov), nrow(centered_data) + 1)

  for (i in 1:(length(segment_points) - 1)) {
    segment_length <- segment_points[i + 1] - segment_points[i]
    data_segmented <- given_data[(segment_points[i]):(segment_points[i + 1] - 1),]
    nws_mean <- nws[2 * nws <= segment_length]
    if (length(nws_mean)>0) {
      res_mean <- mxPBF_mean(data_segmented, nws_mean, alps, FPR_want, n_sample, n_cores)
      res_mean_list <- c(res_mean_list, setNames(list(res_mean), paste(segment_points[i], "to", segment_points[i + 1])))
      changes_mean <- append(changes_mean, majority_rule_mxPBF(res_mean, nws_mean, n) + segment_points[i] - 1)
    }
  }
  mxPBF_result <- list("Result_cov" = res_cov,
                       "Result_mean" = res_mean_list,
                       "Change_points_cov" = sort(unlist(changes_cov)),
                       "Change_points_mean" = sort(unlist(changes_mean)))
  return(mxPBF_result)
}
