#' Fit Vector Autoregressive Model using Mplus
#'
#' This function estimates the parameters of a VAR model
#' using `Mplus`.
#'
#' @author Ivan Jacob Agaloos Pesigan
#'
#' @param data Numeric matrix.
#'   The time series data with dimensions `t` by `k`,
#'   where `t` is the number of observations
#'   and `k` is the number of variables.
#' @param p Positive integer.
#'   Number of lags.
#' @param mplus_bin Character string.
#'   Path to the Mplus binary.
#' @param iter Positive integer.
#'   Maximum number of iterations
#'   for each MCMC chain.
#'
#' @examples
#' \dontrun{
#' FitVARMplus(data = dat_p1, p = 1, mplus_bin = "mpdemo")
#' FitVARMplus(data = dat_p2, p = 2, mplus_bin = "mpdemo")
#' }
#'
#' @family Fitting Autoregressive Model Functions
#' @keywords fitAutoReg fit
#' @export
FitVARMplus <- function(data,
                        p = 1,
                        mplus_bin,
                        iter = 5000L) {
  # get dims
  dims_y <- dim(data)
  time <- dims_y[1]
  k <- dims_y[2]
  # set temp directory and file names
  wd <- getwd()
  on.exit(
    expr = setwd(dir = wd),
    add = TRUE
  )
  tmpdir <- tempdir()
  on.exit(
    expr = unlink(
      x = tmpdir
    ),
    add = TRUE
  )
  setwd(dir = tmpdir)
  tempfile <- tempfile(
    pattern = "mplus_",
    tmpdir = tmpdir
  )
  fn_inp <- paste0(basename(tempfile), ".inp")
  fn_out <- paste0(basename(tempfile), ".out")
  fn_data <- paste0(basename(tempfile), ".csv")
  fn_res <- paste0(basename(tempfile), ".res")
  # process data
  data[is.na(data)] <- -999
  data <- cbind(
    data,
    seq_len(time)
  )
  utils::write.table(
    x = data,
    file = file.path(tmpdir, fn_data),
    row.names = FALSE,
    col.names = FALSE,
    sep = ","
  )
  # mplus input
  y_names <- paste0("Y", seq_len(k))
  model <- paste0(
    "VAR(k = ",
    k,
    ", p = ",
    p,
    ")"
  )
  title <- paste0("TITLE:\t", model, ";\n")
  dat <- paste0("DATA:\n\tFILE = ", fn_data, ";\n")
  var_names <- paste0(
    "\tNAMES = ",
    paste0(y_names, collapse = " "),
    " TIME",
    ";\n"
  )
  var_usevar <- paste0(
    "\tUSEVARIABLES = ",
    paste0(y_names, collapse = " "),
    ";\n"
  )
  var_lagged <- paste0(
    "\tLAGGED = ",
    paste0(
      paste0(y_names, "(", p, ")"),
      collapse = " "
    ),
    ";\n"
  )
  var_tinterval <- "\tTINTERVAL = TIME(1);\n"
  var_missing <- "\tMISSING = ALL (-999);\n"
  variable <- paste0(
    "VARIABLE:\n",
    var_names,
    var_usevar,
    var_lagged,
    var_tinterval,
    var_missing
  )
  analysis_estimator <- "\tESTIMATOR = BAYES;\n"
  analysis_biterations <- paste0("\tBITERATIONS = (", iter, ");\n")
  analysis <- paste0(
    "ANALYSIS:\n",
    analysis_estimator,
    analysis_biterations
  )
  model <- do.call(
    what = "rbind",
    args = lapply(
      X = seq_len(p),
      FUN = function(i) {
        do.call(
          what = "cbind",
          args = lapply(
            X = seq_len(k),
            FUN = function(j) {
              return(
                paste0("\tY", j, " ON ", "Y", seq_len(k), "&", i, ";\n")
              )
            }
          )
        )
      }
    )
  )
  dim(model) <- NULL
  model <- paste0(
    "MODEL:\n",
    paste0(model, collapse = "")
  )
  output <- paste0(
    "OUTPUT:\n",
    "\tTECH8;\n"
  )
  savedata <- paste0(
    "SAVEDATA:\n",
    "\tRESULTS = ", fn_res, ";\n"
  )
  input <- paste0(
    title,
    dat,
    variable,
    analysis,
    model,
    output,
    savedata
  )
  writeLines(
    text = input,
    con = file.path(tmpdir, fn_inp)
  )
  # run
  system(
    paste(
      mplus_bin,
      fn_inp,
      fn_out
    ),
    ignore.stdout = TRUE,
    ignore.stderr = TRUE
  )
  return(
    list(
      output = readLines(
        con = file.path(tmpdir, fn_out)
      ),
      results = readLines(
        con = file.path(tmpdir, fn_res)
      )
    )
  )
}
