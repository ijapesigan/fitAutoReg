#' Parametric Bootstrap Confidence Intervals
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param x Numeric matrix.
#'   Output of [PBootVAROLS()].
#' @param alpha Numeric.
#'   Significance level.
#'
#' @return A list with two elements, namely `ll` for the lower limit
#'   and `ul` for the upper limit.
#'
#' @examples
#' set.seed(42)
#' system.time(pb <- PBootVAROLS(data = dat_p2, p = 2, B = 10, burn_in = 20))
#' pb$est
#' PBootCI(pb)
#' system.time(pb <- PBootVARLasso(
#'   data = dat_p2, p = 2, B = 10, burn_in = 20,
#'   n_lambdas = 100, crit = "ebic", max_iter = 1000, tol = 1e-5
#' ))
#' pb$est
#' PBootCI(pb)
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords fitAutoReg pb
#' @export
PBootCI <- function(x, alpha = 0.05) {
  ll <- alpha / 2
  ul <- 1 - alpha / 2
  q <- dim(x$boot)[2]
  output <- matrix(
    data = 0.0,
    nrow = q,
    ncol = 2
  )
  colnames(output) <- c(ll * 100, ul * 100)
  for (i in seq_len(q)) {
    output[i, ] <- stats::quantile(x$boot[, i], probs = c(ll, ul))
  }
  dims <- dim(x$est)
  return(
    list(
      ll = matrix(
        data = output[, 1],
        nrow = dims[1],
        ncol = dims[2]
      ),
      ul = matrix(
        data = output[, 2],
        nrow = dims[1],
        ncol = dims[2]
      )
    )
  )
}
