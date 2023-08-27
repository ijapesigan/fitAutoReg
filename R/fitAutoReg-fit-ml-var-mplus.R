#' Fit Multilevel Vector Autoregressive Model using Mplus
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
#' FitMLVARMplus(data = dat_ml_p1, p = 1, mplus_bin = "mpdemo")
#' FitMLVARMplus(data = dat_ml_p2, p = 2, mplus_bin = "mpdemo")
#' }
#'
#' @family Fitting Autoregressive Model Functions
#' @keywords fitAutoReg fit
#' @export
FitMLVARMplus <- function(data,
                          p = 1,
                          mplus_bin,
			               iter = 5000L) {
  # get dims
  dims_y <- dim(data[[1]])
  n <- length(data)
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
  data <- do.call(
    what = "rbind",
    args = lapply(
      X = seq_len(n),
      FUN = function(i) {
        data <- data[[i]]
        data[is.na(data)] <- -999
        data <- cbind(
          data,
          seq_len(time),
          i
        )
      }
    )
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
    "ML-VAR(k = ",
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
    " TIME ID",
    ";\n"
  )
  var_usevar <- paste0(
    "\tUSEVARIABLES = ",
     paste0(y_names, collapse = " "),
    ";\n"
  )
  var_cluster <- "\tCLUSTER = ID;\n"
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
    var_cluster,
	var_lagged,
	var_tinterval,
	var_missing
  )
  analysis_type <- "\tTYPE = TWOLEVEL RANDOM;\n"
  analysis_estimator <- "\tESTIMATOR = BAYES;\n"
  analysis_biterations <- paste0("\tBITERATIONS = (", iter, ");\n")
  analysis <- paste0(
    "ANALYSIS:\n",
    analysis_type,
	analysis_estimator,
	analysis_biterations
  )
  model_within <- do.call(
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
                paste0("Y", j, " ON ", "Y", seq_len(k), "&", i, ";\n")
              )
            }
          )
        )
      }
    )
  )
  dim(model_within) <- NULL
  beta <- paste0(
    "B",
    seq_len(length(model_within))
  )
  error <- paste0("LN", y_names)
  model_within <- mapply(
    FUN = function(beta,
                   model_within) {
      return(
        paste0("\t\t", beta , " | ", model_within)
      )
    },
    beta = beta,
    model_within = model_within
  )
  model_within <- c(
    model_within,
    paste0("\t\t", error, " | ", y_names, ";\n")
  )
  model_within <- paste0(
    "\t%WITHIN%\n",
    paste0(model_within, collapse = "")
  )
  model_between <- c(
    paste0(
      "\t\t",
      "[",
      y_names,
      "]",
      ";\n",
      "\t\t",
      y_names,
      ";\n"
    ),
    paste0(
      "\t\t",
      "[",
      error,
      "]",
      ";\n",
      "\t\t",
      error,
      ";\n"
    ),
    paste0(
      "\t\t",
      "[",
      beta,
      "]",
      ";\n",
      "\t\t",
      beta,
      ";\n"
    )
  )
  model_between <- paste0(
    "\t%BETWEEN%\n",
    paste0(model_between, collapse = "")
  )
  model <- paste0(
    "MODEL:\n",
    model_within,
    model_between
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
