#' Parametric Bootstrap Standard Errors
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param x Numeric matrix.
#'   Output of [PBootVAROLS()].
#'
#' @return A matrix of standard error.
#'
#' @examples
#' set.seed(42)
#' system.time(pb <- PBootVAROLS(data = dat_p2, p = 2, B = 10, burn_in = 20))
#' pb$est
#' PBootSE(pb)
#' system.time(pb <- PBootVARLasso(
#'   data = dat_p2, p = 2, B = 10, burn_in = 20,
#'   n_lambdas = 100, crit = "ebic", max_iter = 1000, tol = 1e-5
#' ))
#' pb$est
#' PBootSE(pb)
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords fitAutoReg pb
#' @export
PBootSE <- function(x) {
  dims <- dim(x$est)
  return(
    matrix(
      data = sqrt(diag(stats::var(x$boot))),
      nrow = dims[1],
      ncol = dims[2]
    )
  )
}
