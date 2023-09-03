#' Data Analysis - `data/dat_p2_yx.rda`
#'
DataProcessDatP2YX <- function() {
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
  dat_p2 <- simAutoReg::SimVAR(
    time = time,
    burn_in = burn_in,
    constant = constant,
    coef = coef,
    chol_cov = chol_cov
  )
  dat_p2_yx <- simAutoReg::YX(dat_p2, p)
  save(
    dat_p2_yx,
    file = file.path(
      data_dir,
      "dat_p2_yx.rda"
    ),
    compress = "xz"
  )
}
DataProcessDatP2YX()
rm(DataProcessDatP2YX)
