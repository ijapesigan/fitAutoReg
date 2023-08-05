#' Select the Lasso Estimates from the Grid Search
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param search Object.
#'   Output of the [SearchVARLasso()] function.
#' @param crit Character string.
#'   Information criteria to use.
#'   Valid values include `"aic"`, `"bic"`, and `"ebic"`.
#' @return Returns the Lasso estimates
#' of autoregression and cross regression coefficients.
#'
#' @examples
#' Ystd <- StdMat(dat_demo_yx$Y)
#' Xstd <- StdMat(dat_demo_yx$X[, -1])
#' lambdas <- 10^seq(-5, 5, length.out = 100)
#' search <- SearchVARLasso(
#'   Ystd = Ystd, Xstd = Xstd, lambdas = lambdas,
#'   max_iter = 10000, tol = 1e-5
#' )
#' SelectVARLasso(search, crit = "ebic")
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords fitAutoReg fit
#' @export
SelectVARLasso <- function(search, crit = "ebic") {
  stopifnot(crit %in% c("aic", "bic", "ebic"))
  info <- search$criteria
  info <- cbind(info, seq_len(nrow(info)))
  if (crit == "aic") {
    y <- info[order(info[, 2], decreasing = FALSE), ]
  }
  if (crit == "bic") {
    y <- info[order(info[, 3], decreasing = FALSE), ]
  }
  if (crit == "ebic") {
    y <- info[order(info[, 4], decreasing = FALSE), ]
  }
  beta <- search$fit[[y[1, 5]]]
  attr(beta, "lambda") <- y[1, 1]
  attr(beta, "aic") <- y[1, 2]
  attr(beta, "bic") <- y[1, 3]
  attr(beta, "ebic") <- y[1, 4]
  return(beta)
}
