#' Data Analysis - `data/dat_demo_exo_yx.rda`
#'
DataAnalysisDatDemoExoYX <- function() {
  rproj <- rprojroot::is_rstudio_project
  data_dir <- rproj$find_file(
    "data"
  )
  dir.create(
    path = data_dir,
    showWarnings = FALSE,
    recursive = TRUE
  )
  set.seed(42)
  time <- 1000L
  burn_in <- 200
  k <- 3
  p <- 2
  constant <- c(1, 1, 1)
  coef <- matrix(
    data = c(
      0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
      0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
      0.0, 0.0, 0.6, 0.0, 0.0, 0.3
    ),
    nrow = k,
    byrow = TRUE
  )
  chol_cov <- chol(diag(3))
  exo_mat <- simAutoReg::SimMVN(
    n = time + burn_in,
    location = c(0, 0, 0),
    chol_scale = chol(diag(3))
  )
  exo_coef = matrix(
    data = c(
      0.5, 0.0, 0.0,
      0.0, 0.5, 0.0,
      0.0, 0.0, 0.5
    ),
    nrow = 3
  )
  dat_demo_exo_yx <- simAutoReg::SimVARExo(
    time = time,
    burn_in = burn_in,
    constant = constant,
    coef = coef,
    chol_cov = chol_cov,
    exo_mat = exo_mat,
    exo_coef = exo_coef
  )
  dat_demo_exo_yx <- simAutoReg::YXExo(
    data = dat_demo_exo_yx,
    p = p,
    exo_mat = exo_mat[(burn_in + 1):(burn_in + time), , drop = FALSE]
  )
  save(
    dat_demo_exo_yx,
    file = file.path(
      data_dir,
      "dat_demo_exo_yx.rda"
    ),
    compress = "xz"
  )
}
DataAnalysisDatDemoExoYX()
rm(DataAnalysisDatDemoExoYX)
