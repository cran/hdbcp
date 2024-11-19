#' Majority Rule for Multiscale approach using mxPBF Results
#'
#' This function implements a majority rule-based post-processing approach to identify common change points across multiple window sizes from mxPBF results.
#'
#' @param res_mxPBF A list of results from \code{mxPBF_mean()} or \code{mxPBF_cov()}.
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
#' majority_rule_mxPBF(res_mxPBF)
#' }
#'
#' @export
majority_rule_mxPBF <- function(res_mxPBF) {
  num_windows <- length(res_mxPBF)
  majority_criterion <- num_windows / 2
  selected_group_list <- list()
  for (nw in 1:(floor(num_windows/2) + 1)) {
    window <- res_mxPBF[[nw]]$Window_size
    cps <- res_mxPBF[[nw]]$Change_points
    cps <- cps[cps %notin% unlist(selected_group_list)] # Filter out the used points
    if (length(cps)>0) {
      candidate_group_list <- list()
      for (i in 1:length(cps)) {
        point <- cps[i]
        interval <- c(point - window + 1, point + window - 1)
        candidate_group <- numeric()
        for (w in 1:num_windows) {
          cps_in_wider_nws <- res_mxPBF[[w]]$Change_points
          cps_in_wider_nws <- cps_in_wider_nws[cps_in_wider_nws %notin% unlist(selected_group_list)] # Filter out the used points
          cps_within_interval <- cps_in_wider_nws[(cps_in_wider_nws >= interval[1] & cps_in_wider_nws <= interval[2])]
          candidate_group <- c(candidate_group, cps_within_interval)
        }
        candidate_group_list[[i]] <- candidate_group
      }
      for (i in 1:length(candidate_group_list)) {
        most_interval <- which(sapply(candidate_group_list, length) == max(sapply(candidate_group_list, length)))
        if (max(sapply(candidate_group_list, length)) >= majority_criterion) {
          if (length(most_interval) > 1) {
            most_interval <- most_interval[which.min(sapply(candidate_group_list[most_interval], sample_variance))]
          }
          selected_group <- candidate_group_list[[most_interval]]
          selected_group_list <- append(selected_group_list, list(selected_group))
          candidate_group_list[[most_interval]] <- NULL
        }
      }
    }
  }
  if (length(selected_group_list) > 0) {
    return(sort(sapply(selected_group_list, function(x) round(mean(x)))))
  }
  return(numeric(0))
}
