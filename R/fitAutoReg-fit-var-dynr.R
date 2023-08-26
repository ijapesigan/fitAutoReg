#' Fit Vector Autoregressive (VAR(p = 2)) Model using dynr
#'
#' This function estimates the parameters of a VAR(p = 2) model
#' using the `dynr` package.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param model Ouput of [ModelVARP1Dynr()] or [ModelVARP2Dynr()].
#' @param conf_level a  cumulative proportion indicating
#'   the level of desired confidence intervals
#'   for the final parameter estimates (default is .95)
#' @inheritParams dynr::dynr.cook
#'
#' @return Object of class `dynrCook`.
#'
#' @examples
#' \dontrun{
#' FitVARDynr(model = ModelVARP2Dynr(data = dat_demo))
#' }
#'
#' @family Fitting Autoregressive Model Functions
#' @keywords fitAutoReg fit
#' @export
FitVARDynr <- function(model,
                       conf_level = 0.95,
                       optimization_flag = TRUE,
                       hessian_flag = TRUE,
                       verbose = FALSE,
                       weight_flag = FALSE,
                       debug_flag = FALSE,
                       perturb_flag = FALSE) {
  if (verbose) {
    return(
      dynr::dynr.cook(
        dynrModel = model[["model"]],
        conf.level = conf_level,
        optimization_flag = optimization_flag,
        hessian_flag = hessian_flag,
        verbose = verbose,
        weight_flag = weight_flag,
        debug_flag = debug_flag,
        perturb_flag = perturb_flag
      )
    )
  } else {
    return(
      utils::capture.output(
        dynr::dynr.cook(
          dynrModel = model[["model"]],
          conf.level = conf_level,
          optimization_flag = optimization_flag,
          hessian_flag = hessian_flag,
          verbose = verbose,
          weight_flag = weight_flag,
          debug_flag = debug_flag,
          perturb_flag = perturb_flag
        )
      )
    )
  }
}
