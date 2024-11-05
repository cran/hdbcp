#' Generate Simulated Datasets with Change Points in Mean Vector
#'
#' This function generates simulated datasets that include change points in the mean vector for change point detection.
#' Users can specify various parameters to control the dataset size, dimension, size of signal, and change point locations.
#' The generated datasets include datasets with and without change points, allowing for comparisons in simulation studies.
#'
#' @param n Number of observations to generate.
#' @param p Number of features or dimensions for each observation.
#' @param signal_size Magnitude of the signal to apply at change points.
#' @param pre_proportion Proportion of the covariance matrix's off-diagonal elements to be set to a pre-defined value (default is 0.4).
#' @param pre_value Value assigned to selected off-diagonal elements of the covariance matrix (default is 0.3).
#' @param single_point Location of a single change point in the dataset (default is n/2).
#' @param multiple_points Locations of multiple change points within the dataset (default is quartiles of n).
#' @param type Integer specifying the type of dataset to return. Options are as follows:
#'   - 1: No change points (H0 data)
#'   - 2: Single change point with rare signals
#'   - 3: Single change point with many signals
#'   - 4: Multiple change points with rare signals
#'   - 5: Multiple change points with many signals
#'   The default options are 1, 2, 3, 4, and 5.
#'
#' @return A 3D array containing the generated datasets. Each slice represents a different dataset type.
#'
#' @examples
#' # Generate a default dataset
#' datasets <- generate_mean_datasets(100, 50, 1)
#'
#' null_data <- datasets[,,1]
#' single_many_data <- datasets[,,3]
#'
#' @export
generate_mean_datasets <- function(n = 500, p = 200, signal_size = 1, pre_proportion = 0.4, pre_value = 0.3, single_point = round(n/2), multiple_points = c(round(n/4), round(2*n/4), round(3*n/4)), type = c(1,2,3,4,5)) {
  generate_mean_data <- function(n, mu, mu2, cov, signal_loc) {
    signal_loc <- sort(signal_loc)
    start <- 1
    data_list <- list()
    for (i in seq_along(signal_loc)) {
      end <- signal_loc[i]
      if (i %% 2 == 1) {
        data_list[[i]] <- cpp_mvrnorm(end - start + 1, mu, sigma)
      } else {
        data_list[[i]] <- cpp_mvrnorm(end - start + 1, mu2, sigma)
      }
      start <- end + 1
    }
    if (start <= n) {
      if (length(signal_loc) %% 2 == 1) {
        data_list[[length(signal_loc) + 1]] <- cpp_mvrnorm(n - start + 1, mu2, sigma)
      } else {
        data_list[[length(signal_loc) + 1]] <- cpp_mvrnorm(n - start + 1, mu, sigma)
      }
    }
    data <- do.call(rbind, data_list)

    return(data)
  }
  # Initialize the mean vector
  mu <- rep(0, p)
  mu_R <- mu_M <- mu

  # Set the indices where the signal will be applied
  indices_R <- sample(length(mu), 5)
  indices_M <- sample(length(mu), p / 2)

  # Add signal to the mean vectors
  mu_R[indices_R] <- signal_size
  mu_M[indices_M] <- signal_size

  # Create the Omega matrix
  omega <- matrix(0, p, p)
  lower_tri_indices <- which(lower.tri(omega))
  selected_indices <- sample(lower_tri_indices, floor(pre_proportion * length(lower_tri_indices)))
  omega[selected_indices] <- pre_value
  omega <- omega + t(omega)
  diag(omega) <- diag(omega) / 2

  # Adjust the Omega matrix if the minimum eigenvalue is too small
  min_eigen_val <- min(eigen(omega)$values)
  if (min_eigen_val < 1e-5) {
    omega <- omega + (-min_eigen_val + 0.1^3) * diag(1, p)
  }

  # Calculate the Sigma matrix
  sigma <- solve(omega)

  # Generate data
  sim_data_H0 <- cpp_mvrnorm(n, mu, sigma)
  sim_data_H1R_single <- generate_mean_data(n, mu, mu_R, sigma, single_point)
  sim_data_H1M_single <- generate_mean_data(n, mu, mu_M, sigma, single_point)
  sim_data_H1R_multiple <- generate_mean_data(n, mu, mu_R, sigma, multiple_points)
  sim_data_H1M_multiple <- generate_mean_data(n, mu, mu_M, sigma, multiple_points)

  # Return the results as an array
  given_datasets <- array(c(sim_data_H0, sim_data_H1R_single, sim_data_H1M_single,
                            sim_data_H1R_multiple, sim_data_H1M_multiple),
                          dim = c(dim(sim_data_H0), 5))

  return(given_datasets[, , type])
}
