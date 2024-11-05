#' Majority Rule for Multiscale approach using mxPBF Results
#'
#' This function implements a majority rule-based post-processing approach to identify common change points across multiple window sizes from mxPBF results.
#'
#' @param result_mxPBFs A list of results from \code{mxPBF_mean()} or \code{mxPBF_cov()}.
#' @param nws A vector of window sizes used for \code{mxPBF_mean()} or \code{mxPBF_cov()}.
#' @param n The total number of observations in the dataset.
#'
#' @return A vector of final detected change points that are common across multiple windows based on majority rule.
#'
#'
#' @examples
#' \donttest{
#' n <- 500
#' p <- 200
#' signal_size <- 1
#' pre_value <- 0.3
#' pre_proportion <- 0.4
#' given_data <- generate_mean_datasets(n, p, signal_size, pre_proportion, pre_value,
#' single_point = 250, multiple_points = c(150,300,350), type = 5)
#' nws <- c(25, 60, 100)
#' alps <- seq(1,10,0.05)
#' res_mxPBF <- mxPBF_mean(given_data, nws, alps)
#' majority_rule_mxPBF(res_mxPBF, nws, n)
#' }
#'
#' @export
majority_rule_mxPBF <- function(result_mxPBFs, nws, n) {
  post_process <- function(detected_points_list, nw) {
    W <- length(detected_points_list)
    groups <- list()
    for (w in 1:W) {
      points <- detected_points_list[[w]]
      points <- points[points %notin% unlist(groups)]
      while (length(points)>0) {
        grouping <- sapply(seq_along(points), function(i){
          point <- points[i]
          interval <- (point - nw + 1):(point + nw - 1)
          interval <- interval[interval %notin% unlist(groups)]
          points_in_group <- unlist(sapply(detected_points_list[w:W], function(x) x[x %in% interval]))
          return(list(len = length(points_in_group), points = points_in_group))
        })
        if (max(unlist(grouping[1,])) <= 1) {
          for (i in 1:ncol(grouping)) {
            groups <- append(groups, grouping[2,i])
          }
          break
        }
        most_interval <- which(unlist(grouping[1,]) == max(unlist(grouping[1,])))
        if (length(most_interval)>1) {
          variances <- sapply(most_interval, function(i) var(unlist(grouping[2,i])))
          most_interval <- most_interval[which.min(variances)]
        }
        groups <- append(groups,grouping[2,most_interval])
        points <- points[points %notin% unlist(groups)]
      }
    }
    return(groups)
  }
  pre_groups <- lapply(result_mxPBFs, function(res) {
    res$Change_points
  })
  post_groups <- post_process(pre_groups, nws[1])
  major_groups <- post_groups[unlist(lapply(post_groups, function(x) length(x) >= length(pre_groups)/2))]
  minor_groups <- post_groups[unlist(lapply(post_groups, function(x) length(x) < length(pre_groups)/2))]
  for (i in seq_along(minor_groups)) {
    loc <- mean(unlist(minor_groups[[i]]))
    criteria <- length(pre_groups)
    while(loc < nws[criteria] || loc >= n - nws[criteria]) {
      if (criteria == 1) {
        break
      }
      criteria = criteria - 1
    }
    if (length(unlist(minor_groups[[i]])) >= criteria / 2) {
      major_groups <- append(major_groups, minor_groups[[i]])
    }
  }
  if (length(major_groups) == 0) {
    return(integer(0))
  }
  return(unlist(lapply(major_groups, function(x) round(mean(x)))))
}
