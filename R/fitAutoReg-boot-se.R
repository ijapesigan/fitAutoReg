#' Bootstrap Standard Errors
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param x Numeric matrix.
#'   Output of
#'   [PBootVAROLS()],
#'   [PBootVARLasso()],
#'   [RBootVAROLS()], or
#'   [RBootVARLasso()].
#'
#' @return A matrix of standard errors.
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
#' BootSE(pb)
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
#' BootSE(pb)
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
#' BootSE(rb)
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
#' BootSE(rb)
#'
#' @family Simulation of Autoregressive Data Functions
#' @keywords fitAutoReg pb rb
#' @export
BootSE <- function(x) {
  dims <- dim(x$est)
  return(
    matrix(
      data = sqrt(diag(stats::var(x$boot))),
      nrow = dims[1],
      ncol = dims[2]
    )
  )
}
