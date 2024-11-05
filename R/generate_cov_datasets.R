#' Generate Simulated Datasets with Change Points in Covariance Matrix
#'
#' This function generates simulated datasets that include change points in the covariance matrix for change point detection.
#' Users can specify various parameters to control the dataset size, dimension, size of signal, and change point locations.
#' The generated datasets include datasets with and without change points, allowing for comparisons in simulation studies.
#'
#' @param n Number of observations to generate.
#' @param p Number of features or dimensions for each observation.
#' @param signal_size Magnitude of the signal applied at change points.
#' @param sparse Determines if a sparse covariance structure is used (default is TRUE).
#' @param single_point Location of a single change point in the dataset (default is n/2).
#' @param multiple_points Locations of multiple change points within the dataset (default is quartiles of n).
#' @param type Integer vector specifying the type of dataset to return. Options are as follows:
#'   - 1: No change points (H0 data)
#'   - 2: Single change point with rare signals
#'   - 3: Single change point with many signals
#'   - 4: Multiple change points with rare signals
#'   - 5: Multiple change points with many signals
#'
#' @return A 3D array containing the generated datasets. Each slice represents a different dataset type.
#'
#' @examples
#' # Generate a default dataset
#' datasets <- generate_cov_datasets(100, 50, 1)
#'
#' null_data <- datasets[,,1]
#' single_many_data <- datasets[,,3]
#'
#' @export
generate_cov_datasets <- function(n, p, signal_size, sparse = TRUE, single_point = round(n/2), multiple_points = c(round(n/4), round(2*n/4), round(3*n/4)), type = c(1,2,3,4,5)) {
  gen_U <- function(p, signal, rare = TRUE) {
    if (rare) {
      U <- matrix(0, nrow = p, ncol = p)

      lower_tri_indices <- which(lower.tri(U ,diag = T))
      selected_indices <- sample(lower_tri_indices, 5)

      U[selected_indices] <- runif(5, 0, signal)
      U <- U + t(U)
      diag(U) <- diag(U) / 2
    } else {
      u <- runif(p, 0, signal)
      U <- u %*% t(u)
    }
    return(U)
  }
  gen_Sigma <- function(p, sparse = TRUE) {
    if (sparse) {
      Delta_1 <- matrix(0, nrow = p, ncol = p)
      lower_tri_indices <- which(lower.tri(Delta_1 ,diag = T))
      selected_indices <- sample(lower_tri_indices, floor(0.05 * length(lower_tri_indices)))
      Delta_1[selected_indices] <- 0.5
      Delta_1 <- Delta_1 + t(Delta_1)
      diag(Delta_1) <- diag(Delta_1) / 2
      min_eigenvalue <- min(eigen(Delta_1)$values)
      if (min_eigenvalue <= 1e-5) {
        Delta_1 <- Delta_1 + (abs(min_eigenvalue) + 0.05) * diag(p)
      }
      d <- runif(p, 0.5, 2.5)
      D_sqrt <- diag(sqrt(d))
      Sigma_1 <- D_sqrt %*% Delta_1 %*% D_sqrt
    } else {
      omega <- runif(p, 1, 5)
      Omega <- diag(omega)
      Delta <- matrix(0, nrow = p, ncol = p)
      for (i in 1:p) {
        for (j in 1:p) {
          Delta[i, j] <- (-1)^(i + j) * 0.4^(abs(i - j)^(1/10))
        }
      }
      Sigma_1 <- Omega %*% Delta %*% Omega
    }
    return(Sigma_1)
  }
  generate_cov_data <- function(n, mu, sigma, sigma2, signal_loc) {
    signal_loc <- sort(signal_loc)
    start <- 1
    data_list <- list()
    for (i in seq_along(signal_loc)) {
      end <- signal_loc[i]
      if (i %% 2 == 1) {
        data_list[[i]] <- cpp_mvrnorm(end - start + 1, mu, sigma)
      } else {
        data_list[[i]] <- cpp_mvrnorm(end - start + 1, mu, sigma2)
      }
      start <- end + 1
    }
    if (start <= n) {
      if (length(signal_loc) %% 2 == 1) {
        data_list[[length(signal_loc) + 1]] <- cpp_mvrnorm(n - start + 1, mu, sigma2)
      } else {
        data_list[[length(signal_loc) + 1]] <- cpp_mvrnorm(n - start + 1, mu, sigma)
      }
    }
    data <- do.call(rbind, data_list)

    return(data)
  }
  mu <- rep(0,p)
  sigma <- gen_Sigma(p,sparse)
  U_rare <- gen_U(p,signal_size,rare=T)
  U_many <- gen_U(p,signal_size,rare=F)
  sigma_R <- sigma + U_rare
  sigma_M <- sigma + U_many
  min_eigen1 <- min(eigen(sigma)$values)
  min_eigen2 <- min(eigen(sigma_R)$values)
  min_eigen3 <- min(eigen(sigma_M)$values)
  if (any(c(min_eigen1,min_eigen2,min_eigen3) < 1e-5)) {
    delta <- abs(min(c(min_eigen1,min_eigen2,min_eigen3))) + 0.05
    sigma <- sigma + delta * diag(p)
    sigma_R <- sigma_R + delta * diag(p)
    sigma_M <- sigma_M + delta * diag(p)
  }

  # Generate data
  sim_data_H0 <- cpp_mvrnorm(n, mu, sigma)
  sim_data_H1R_single <- generate_cov_data(n, mu, sigma, sigma_R, single_point)
  sim_data_H1M_single <- generate_cov_data(n, mu, sigma, sigma_M, single_point)
  sim_data_H1R_multiple <- generate_cov_data(n, mu, sigma, sigma_R, multiple_points)
  sim_data_H1M_multiple <- generate_cov_data(n, mu, sigma, sigma_M, multiple_points)

  # Return the results as an array
  given_datasets <- array(c(sim_data_H0, sim_data_H1R_single, sim_data_H1M_single,
                            sim_data_H1R_multiple, sim_data_H1M_multiple),
                          dim = c(dim(sim_data_H0), 5))

  return(given_datasets[, , type])
}
