`%notin%` <- Negate(`%in%`)

# Function to calculate the sample variance, even when there is only one sample.
sample_variance <- function(x) {
  if (length(x) <= 1) {
    return(0)
  } else {
    return(var(x))
  }
}

# Function to apply centering using a sliding window technique.
sliding_window_centering <- function(x, window_size) {
  n <- length(x)
  half_window <- floor(window_size / 2)

  centered <- rep(NA, n)

  for (i in 1:n) {
    start_idx <- max(1, i - half_window)
    end_idx <- min(n, i + half_window)

    window_mean <- mean(x[start_idx:end_idx], na.rm = TRUE)
    centered[i] <- x[i] - window_mean
  }
  return(centered)
}
