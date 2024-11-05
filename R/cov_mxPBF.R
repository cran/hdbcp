#' Change Point Detection in Covaraiance Structure using Maximum Pairwise Bayes Factor (mxPBF)
#'
#' This function detects change points in the covariance structure of multivariate Gaussian data using the Maximum Pairwise Bayes Factor (mxPBF).
#' The function selects alpha that controls the empirical False Positive Rate (FPR), as suggested in the paper.
#' One can conduct a multiscale approach using the function \code{majority_rule_mxPBF()}.
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
#'
#' @return A list of length equal to the number of window sizes provided. Each element in the list contains: \describe{
#' \item{Change_points}{Locations of detected change points.}
#' \item{Bayes_Factors}{Vector of calculated Bayes Factors for each middle points.}
#' \item{Selected_alpha}{Optimal alpha value selected based on the method that controls the empirical FPR.}
#' \item{Window_size}{Window size used for change point detection.}
#' }
#'
#' @examples
#' \donttest{
#' nws <- c(25, 60, 100)
#' alps <- seq(1,10,0.05)
#' ## H0 data
#' mu <- rep(0,10)
#' sigma1 <- diag(10)
#' X <- mvrnorm_cpp(500,mu,sigma1)
#' res1 <- mxPBF_cov(X, nws = nws, alps = alps)
#'
#' ## H1 data
#' mu <- rep(0,10)
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
#' Y1 <- mvrnorm_cpp(250,mu,sigma1)
#' Y2 <- mvrnorm_cpp(250,mu,sigma2)
#' Y <- rbind(Y1, Y2)
#' res2 <- mxPBF_cov(Y, nws = nws, alps = alps)
#' }
#'
#' @export
mxPBF_cov <- function(given_data, a0 = 0.01, b0 = 0.01, nws, alps, FPR_want = 0.05, n_sample = 300, n_cores = 1, centering = "skip"){
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

  results <- lapply(nws, function(nw) {
    cp <- numeric(0)
    mxPBF_simulated <- simulate_mxPBF_cov(centered_data, a0, b0, nw, alps, n_sample, n_cores)
    alp_selected <- alps[which.min(abs(rowSums(exp(mxPBF_simulated) > 10) / n_sample - FPR_want))]
    bf_given_data <- cpd_cov_mxPBF(centered_data, a0, b0, nw, alp_selected, n_cores)
    exp_bf <- exp(bf_given_data)
    while (any(exp_bf > 10)) {
      i_tilde <- which(exp_bf > 10)[1]
      i_hat <- i_tilde + which.max(bf_given_data[i_tilde:(i_tilde + nw - 1)]) - 1
      cp <- c(cp, i_hat + nw)
      exp_bf[i_tilde:(i_hat + nw - 1)] <- 0
    }
    mxPBF_result <- list("Change_points" = cp,
                         "Bayes_Factors" = bf_given_data,
                         "Selected_alpha" = alp_selected,
                         "Window_size" = nw)
    class(mxPBF_result) <- "mxPBF"
    return(mxPBF_result)
  })
  names(results) <- paste("Window_size", nws, sep = "_")
  return(results)
}
