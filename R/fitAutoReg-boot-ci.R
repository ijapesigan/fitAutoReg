#' Bootstrap Percentile Confidence Intervals
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param alpha Numeric.
#'   Significance level.
#' @inheritParams BootSE
#'
#' @return A list with two elements, namely `ll` for the lower limit
#'   and `ul` for the upper limit.
#'
#' @examples
#' set.seed(42)
#' # Parametric bootstrap
#' system.time(
#'   pb <- PBootVAROLS(
#'     data = dat_p2,
#'     p = 2,
#'     B = 5,
#'     burn_in = 20
#'   )
#' )
#' pb$est
#' BootCI(pb)
#' system.time(
#'   pb <- PBootVARLasso(
#'     data = dat_p2,
#'     p = 2,
#'     B = 5,
#'     burn_in = 20,
#'     n_lambdas = 50,
#'     crit = "ebic",
#'     max_iter = 1000,
#'     tol = 1e-5
#'   )
#' )
#' pb$est
#' BootCI(pb)
#'
#' # Residual bootstrap
#' system.time(
#'   rb <- RBootVAROLS(
#'     data = dat_p2,
#'     p = 2,
#'     B = 5
#'   )
#' )
#' rb$est
#' BootCI(rb)
#' system.time(
#'   rb <- RBootVARLasso(
#'     data = dat_p2,
#'     p = 2,
#'     B = 5,
#'     n_lambdas = 50,
#'     crit = "ebic",
#'     max_iter = 1000,
#'     tol = 1e-5
#'   )
#' )
#' rb$est
#' BootCI(rb)
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords fitAutoReg pb rb
#' @export
BootCI <- function(x, alpha = 0.05) {
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
