#' Specify Vector Autoregressive (VAR(p = 2)) Model using dynr
#'
#' This function specifies a VAR(p = 2) model
#' using the [dynr::dynr.model()] package.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param data Numeric matrix.
#'   The time series data with dimensions `t` by `k`,
#'   where `t` is the number of observations
#'   and `k` is the number of variables.
#' @inheritParams dynr::dynr.model
#'
#' @examples
#' ModelVARP2Dynr(data = dat_p2)
#'
#' @family Fitting Autoregressive Model Functions
#' @keywords fitAutoReg fit
#' @export
ModelVARP2Dynr <- function(data,
                           outfile = tempfile()) {
  p <- 2
  if (is.list(data)) {
    dims_y <- dim(data[[1]])
  }
  if (is.matrix(data)) {
    dims_y <- dim(data)
  }
  time <- dims_y[1]
  k <- dims_y[2]
  q <- k * p
  y_names <- paste0("y", "_", seq_len(k))
  eta_names <- paste0("eta", "_", seq_len(k))
  dynamics_col <- c(
    paste0(eta_names, "l1"),
    paste0(eta_names, "l2")
  )
  dynamics_row <- c(
    paste0(eta_names),
    paste0(eta_names, "l1")
  )
  dynamics_col_idx <- c(
    paste0(seq_len(k), "l1"),
    paste0(seq_len(k), "l2")
  )
  dynamics_row_idx <- c(
    seq_len(k),
    paste0(seq_len(k), "l1")
  )
  mu_names <- paste0("mu", "_", seq_len(k))
  if (is.list(data)) {
    dims_y <- dim(data[[1]])
    raw_data <- lapply(
      X = seq_len(length(data)),
      FUN = function(i, data, time) {
        x <- cbind(
          data[[i]],
          i,
          seq_len(time)
        )
        colnames(x) <- c(y_names, "id", "time")
        return(x)
      },
      data = data,
      time = time
    )
    raw_data <- as.data.frame(
      do.call(
        what = "rbind",
        args = raw_data
      )
    )
  }
  if (is.matrix(data)) {
    colnames(data) <- y_names
    raw_data <- as.data.frame(data)
    raw_data$id <- 1
    raw_data$time <- seq_len(time)
  }
  # data
  dynr_data <- dynr::dynr.data(
    dataframe = raw_data,
    id = "id",
    time = "time",
    observed = y_names
  )
  params_dyn <- lapply(
    X = dynamics_row_idx[seq_len(k)],
    FUN = function(i) {
      paste0("beta", "_", i, dynamics_col_idx)
    }
  )
  params_dyn <- do.call(
    what = "rbind",
    args = params_dyn
  )
  params_dyn <- rbind(
    params_dyn,
    matrix(
      data = "fixed",
      nrow = nrow(params_dyn),
      ncol = ncol(params_dyn)
    )
  )
  values_dyn <- diag(
    x = 0.00,
    nrow = k,
    ncol = q
  )
  values_dyn <- rbind(
    values_dyn,
    diag(
      x = 1,
      nrow = nrow(values_dyn),
      ncol = ncol(values_dyn)
    )
  )
  colnames(values_dyn) <- colnames(params_dyn) <- dynamics_col
  rownames(values_dyn) <- rownames(params_dyn) <- dynamics_row
  params_int <- c(
    paste0("alpha", "_", seq_len(k)),
    rep(x = "fixed", times = k)
  )
  values_int <- rep(0, times = q)
  dynr_dynamics <- dynr::prep.matrixDynamics(
    params.dyn = params_dyn,
    values.dyn = values_dyn,
    params.int = params_int,
    values.int = values_int,
    isContinuousTime = FALSE
  )
  # measurement
  params_load <- matrix(
    data = "fixed",
    nrow = k,
    ncol = q
  )
  values_load <- diag(
    x = 1,
    nrow = k,
    ncol = q
  )
  state_names <- c(
    paste0(eta_names),
    paste0(eta_names, "l1")
  )
  dynr_measurement <- dynr::prep.measurement(
    values.load = values_load,
    params.load = params_load,
    obs.names = y_names,
    state.names = state_names
  )
  # noise
  x <- seq_len(k)
  params_latent <- matrix(data = 0, k, k)
  for (i in 1:k) {
    for (j in i:k) {
      params_latent[i, j] <- paste0("epsilon", "_", x[i], x[j])
      params_latent[j, i] <- params_latent[i, j]
    }
  }
  params_latent <- cbind(
    params_latent,
    matrix(
      data = "fixed",
      nrow = nrow(params_latent),
      ncol = ncol(params_latent)
    )
  )
  params_latent <- rbind(
    params_latent,
    matrix(
      data = "fixed",
      nrow = nrow(params_latent),
      ncol = ncol(params_latent)
    )
  )
  values_latent <- diag(k)
  values_latent <- cbind(
    values_latent,
    matrix(
      data = 0,
      nrow = nrow(values_latent),
      ncol = ncol(values_latent)
    )
  )
  values_latent <- rbind(
    values_latent,
    matrix(
      data = 0,
      nrow = nrow(values_latent),
      ncol = ncol(values_latent)
    )
  )
  values_observed <- diag(x = 0, nrow = k)
  params_observed <- matrix(
    data = "fixed",
    nrow = k,
    ncol = k
  )
  dynr_noise <- dynr::prep.noise(
    values.latent = values_latent,
    params.latent = params_latent,
    values.observed = values_observed,
    params.observed = params_observed
  )
  # initial state
  x <- seq_len(q)
  params_inicov <- matrix("0", q, q)
  for (i in 1:q) {
    for (j in i:q) {
      params_inicov[i, j] <- paste0("sigma", "_", x[i], x[j])
      params_inicov[j, i] <- params_inicov[i, j]
    }
  }
  rownames(params_inicov) <- colnames(params_inicov) <- state_names
  initial_mu <- c(
    mu_names,
    paste0(mu_names, "l1")
  )
  dynr_initial <- dynr::prep.initial(
    values.inistate = rep(x = 0, times = q),
    params.inistate = initial_mu,
    values.inicov = diag(q),
    params.inicov = params_inicov,
  )
  # model
  out <- list(
    model = dynr::dynr.model(
      dynamics = dynr_dynamics,
      measurement = dynr_measurement,
      noise = dynr_noise,
      initial = dynr_initial,
      data = dynr_data,
      outfile = outfile
    ),
    dynamics = dynr_dynamics,
    measurement = dynr_measurement,
    noise = dynr_noise,
    initial = dynr_initial,
    data = dynr_data,
    dynr_src = readLines(outfile),
    k = k,
    p = p,
    q = q
  )
  class(out) <- c(
    "dynr_model",
    class(out)
  )
  return(out)
}
