#' Data Analysis - `data/dat_p1.rda`
#'
DataProcessDatP1 <- function() {
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
  p <- 1
  constant <- c(1, 1, 1)
  coef <- matrix(
    data = c(
      0.4, 0.0, 0.0,
      0.0, 0.5, 0.0,
      0.0, 0.0, 0.6
    ),
    nrow = k,
    byrow = TRUE
  )
  chol_cov <- chol(diag(3))
  dat_p1 <- simAutoReg::SimVAR(
    time = time,
    burn_in = burn_in,
    constant = constant,
    coef = coef,
    chol_cov = chol_cov
  )
  save(
    dat_p1,
    file = file.path(
      data_dir,
      "dat_p1.rda"
    ),
    compress = "xz"
  )
}
DataProcessDatP1()
rm(DataProcessDatP1)
