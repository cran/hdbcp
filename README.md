# hdbcp
Bayesian Change Point Detection for High-Dimensional Data

## Installation
To install the released version of **hdbcp** from CRAN, use:
```r
install.packages("hdbcp")
```

Alternatively, to install the latest development version from GitHub:
```r
## install.packages("devtools")
## library(devtools)
devtools::install_github("JaeHoonKim98/hdbcp")
```

## Example
You can simply run the function using default parameters.
```r
library(hdbcp)
library(MASS)

nws <- c(25, 60, 100) # A set of window sizes
alps <- seq(1,10,0.05) # A grid of alpha values

# Mean method
mu1 <- rep(0,10)
mu2 <- rep(1,10)
sigma1 <- diag(10)
X <- mvrnorm(500, mu1, sigma1)
Y <- rbind(mvrnorm(250, mu1, sigma1), mvrnorm(250, mu2, sigma1))
res_mxPBF1 <- mxPBF_mean(X, nws, alps)
res_mxPBF2 <- mxPBF_mean(Y, nws, alps)
majority_rule_mxPBF(res_mxPBF1)
majority_rule_mxPBF(res_mxPBF2)

# Covariance method
sigma2 <- diag(10)
for (i in 1:10) {
  for (j in i:10) {
    if (i == j) {
      next
    } else {
      cov_value <- rnorm(1, 1, 1)
      sigma1[i, j] <- cov_value
      sigma1[j, i] <- cov_value
    }
  }
}
Z <- rbind(mvrnorm(250, mu1, sigma1), mvrnorm(250, mu1, sigma2))
res_mxPBF3 <- mxPBF_cov(X, nws, alps)
res_mxPBF4 <- mxPBF_cov(Z, nws, alps)
majority_rule_mxPBF(res_mxPBF3)
majority_rule_mxPBF(res_mxPBF4)

# Combined method
W <- rbind(mvrnorm(150,mu1,sigma1), mvrnorm(150,mu2,sigma1), mvrnorm(200,mu2,sigma2))
mxPBF_combined(W, nws, alps)
```

Alternatively, we can generate data with mean changes and apply the method with specific parameters.
```r
# Set the number of observation, dimension, type of precision matrix, and signals.
n <- 500 # Number of observations
p <- 200 # Dimension
pre_value <- 0.3 # Size of nonzero precision elements
pre_proportion <- 0.4 # Proportion of nonzero precision elements
signal_size <- 1 # Size of signals at change point
single_point <- 250 # Location of change point in single change scenarios
multiple_points <- c(150, 300, 350) # Location of change points in multiple change scenarios

# Set parameters for mxPBF
FPR_want <- 0.05 # Prespecified False Positive Rate when selecting hyperparameter alpha
nws <- c(25, 60, 100)
alps <- seq(1,10,0.05)
n_sample <- 300 # Number of generating samples for calculating empirical FPR
n_cores <- 1 # Number of cores for parellelizing

# Generate dataset with mean changes
given_datasets <- generate_mean_datasets(n, p, signal_size, pre_proportion, pre_value, single_point, multiple_points, type = c(1,2,3,4,5))
## H0 data
given_data <- matrix(given_datasets[,,1], n, p)
res_mxPBF <- mxPBF_mean(given_data, nws, alps, FPR_want, n_sample, n_cores)
majority_rule_mxPBF(res_mxPBF)

## H1 data
given_data <- matrix(given_datasets[,,2], n, p)
res_mxPBF <- mxPBF_mean(given_data, nws, alps, FPR_want, n_sample, n_cores)
majority_rule_mxPBF(res_mxPBF)
```

We can generate data with covariance changes and run the test.
```r
# Set the number of observation, dimension, type of precision matrix, and signals.
n <- 500
p <- 200
sparse <- FALSE # Sparsity of covariance matrix
signal_size <- 3
single_point <- 250
multiple_points <- c(150, 300, 350)

# Set parameters for mxPBF
FPR_want <- 0.05
nws <- c(25, 60, 100)
alps <- seq(1,10,0.05)
n_sample <- 300
n_cores <- 1
a0 <- b0 <- 0.01 # Hyperparameters for mxPBF

# Generate dataset with covariance changes
given_datasets <- generate_cov_datasets(n, p, signal_size, sparse, single_point, multiple_points, type = c(1,2,3,4,5))
## H0 data
given_data <- matrix(given_datasets[,,1], n, p)
res_mxPBF <- mxPBF_cov(given_data, a0, b0, nws, alps, FPR_want, n_sample, n_cores, centering = "skip")
majority_rule_mxPBF(res_mxPBF)

## H1 data
given_data <- matrix(given_datasets[,,2], n, p)
res_mxPBF <- mxPBF_cov(given_data, a0, b0, nws, alps, FPR_want, n_sample, n_cores, centering = "skip")
majority_rule_mxPBF(res_mxPBF)
```
