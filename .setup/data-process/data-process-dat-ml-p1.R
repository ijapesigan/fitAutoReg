#' Data Analysis - `data/dat_ml_p1.rda`
#'
DataProcessDatMLP1 <- function() {
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
  n <- 100L
  time <- 1000L
  burn_in <- 200
  k <- 3
  p <- 1
  constant <- c(1, 1, 1)
  phi <- c(0.4, 0.5, 0.6)
  foo <- function(mu = phi) {
    run <- TRUE
    while (run) {
      coef <- diag(
        x = stats::rnorm(
          n = k,
          mean = mu,
          sd = 0.1
        )
      )
      run <- !simAutoReg::CheckVARCoef(coef)
    }
    return(coef)
  }
  chol_cov <- chol(diag(3))
  dat_ml_p1 <- lapply(
    X = seq_len(n),
    FUN = function(i) {
      return(
        simAutoReg::SimVAR(
          time = time,
          burn_in = burn_in,
          constant = constant,
          coef = foo(),
          chol_cov = chol_cov
        )
      )
    }
  )
  save(
    dat_ml_p1,
    file = file.path(
      data_dir,
      "dat_ml_p1.rda"
    ),
    compress = "xz"
  )
}
DataProcessDatMLP1()
rm(DataProcessDatMLP1)
