#' Print Method for an Object of Class `dynr_model`
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param x Object of class `dynr_model`.
#' @param ... additional arguments.
#'
#' @examples
#' model <- ModelVARP2Dynr(data = dat_demo)
#' print(model)
#'
#' @keywords methods
#' @export
print.dynr_model <- function(x,
                             ...) {
  print(x$model)
}

#' Plot Method for an Object of Class `dynr_model`
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param x Object of class `dynr_model`.
#' @param ... additional arguments.
#'
#' @examples
#' model <- ModelVARP2Dynr(data = dat_demo)
#' plot(model)
#'
#' @keywords methods
#' @export
plot.dynr_model <- function(x,
                            ...) {
  dynr::plotFormula(
    dynrModel = x$model,
    ParameterAs = x$model$param.names,
    printDyn = TRUE,
    printMeas = TRUE
  )
}

#' Auto and Cross Regression Coeffcients
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @return A list with the following elements:
#'   - constant Numeric vector.
#'     The constant term vector of length `k`,
#'     where `k` is the number of variables.
#'   - coef Numeric matrix.
#'     Coefficient matrix with dimensions `k` by `(k * p)`.
#'     Each `k` by `k` block corresponds to the coefficient matrix
#'     for a particular lag.
#'   - cov Numeric matrix.
#'     The `k` by `k` covariance matrix of the noise.
#'
#' @param object Object of class `dynr_model`.
#' @param dynr_cook Ouput of [FitVARDynr()].
#' @param ... additional arguments.
#'
#' @examples
#' \dontrun{
#' dynr_model <- ModelVARP2Dynr(data = dat_demo)
#' dynr_cook <- FitVARP2Dynr(model = dynr_model)
#' coef(object = dynr_model, dynr_cook = dynr_cook)
#' }
#'
#' @keywords methods
#' @export
coef.dynr_model <- function(object,
                            dynr_cook,
                            ...) {
  k <- object$k
  q <- object$q
  params_dyn <- object$dynamics$params.dyn[[1]]
  coefs <- dynr_cook@transformed.parameters
  alpha <- rep(
    x = 0.0,
    times = k
  )
  beta <- matrix(
    data = 0.0,
    nrow = k,
    ncol = q
  )
  for (j in seq_len(q)) {
    for (i in seq_len(k)) {
      beta[i, j] <- coefs[[params_dyn[i, j]]]
    }
  }
  for (i in seq_len(k)) {
    alpha[i] <- coefs[[paste0("alpha", "_", i)]]
  }
  x <- seq_len(k)
  error_cov <- matrix(data = 0, k, k)
  for (i in 1:k) {
    for (j in i:k) {
      error_cov[i, j] <- coefs[[paste0("epsilon", "_", x[i], x[j])]]
      error_cov[j, i] <- error_cov[i, j]
    }
  }
  return(
    list(
      constant = alpha,
      coef = beta,
      cov = error_cov
    )
  )
}
