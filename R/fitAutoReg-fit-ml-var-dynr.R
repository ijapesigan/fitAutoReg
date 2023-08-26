#' Fit Vector Autoregressive (VAR) Model using dynr
#' on Each of the Data Matrix in a List
#'
#' This function estimates the parameters of a VAR model
#' using the `dynr` package
#' for each of the data matrix in a list.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param data List.
#'   Each element is a numeric matrix of
#'   time series data with dimensions `t` by `k`,
#'   where `t` is the number of observations
#'   and `k` is the number of variables.
#' @param p Positive integer.
#'   Number of lags.
#'   Only supports `p = 1` and `p = 2`.
#' @param ncores Positive integer.
#'   Number of cores to use.
#'   Not supported on Windows.
#' @inheritParams FitVARDynr
#'
#' @return A list each element of which
#'   is an object of class `dynrCook`.
#'
#' @family Fitting Autoregressive Model Functions
#' @keywords fitAutoReg fit
#' @export
FitMLVARDynr <- function(data,
                         p,
                         ncores = 1,
                         conf_level = 0.95,
                         optimization_flag = TRUE,
                         hessian_flag = TRUE,
                         verbose = FALSE,
                         weight_flag = FALSE,
                         debug_flag = FALSE,
                         perturb_flag = FALSE) {
  stopifnot(is.list(data))
  stopifnot(p == 1 || p == 2)
  data_1 <- data[[1]]
  dims_y <- dim(data_1)
  time <- dims_y[1]
  k <- dims_y[2]
  y_names <- paste0("y", "_", seq_len(k))
  if (p == 1) {
    model <- ModelVARP1Dynr(data = data_1)
  }
  if (p == 2) {
    model <- ModelVARP2Dynr(data = data_1)
  }
  os <- .OS()
  if (os == "windows") {
    ncores <- 1
  }
  return(
    parallel::mclapply(
      X = data,
      FUN = function(data,
                     model,
                     time,
                     y_names,
                     conf_level,
                     optimization_flag,
                     hessian_flag,
                     verbose,
                     weight_flag,
                     debug_flag,
                     perturb_flag) {
        colnames(data) <- y_names
        raw_data <- as.data.frame(data)
        raw_data$id <- 1
        raw_data$time <- seq_len(time)
        # data
        model$model@data <- dynr::dynr.data(
          dataframe = raw_data,
          id = "id",
          time = "time",
          observed = y_names
        )
        return(
          FitVARDynr(
            model = model,
            conf_level = conf_level,
            optimization_flag = optimization_flag,
            hessian_flag = hessian_flag,
            verbose = verbose,
            weight_flag = weight_flag,
            debug_flag = debug_flag,
            perturb_flag = perturb_flag
          )
        )
      },
      model = model,
      time = time,
      y_names = y_names,
      conf.level = conf_level,
      optimization_flag = optimization_flag,
      hessian_flag = hessian_flag,
      verbose = verbose,
      weight_flag = weight_flag,
      debug_flag = debug_flag,
      perturb_flag = perturb_flag,
      mc.cores = ncores
    )
  )
}
