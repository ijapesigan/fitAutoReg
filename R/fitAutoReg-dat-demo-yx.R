#' Data from the Vector Autoregressive Model (Y) and Lagged Predictors (X)
#'
#' @format A list with elements Y and X where Y is equal to the `dat_demo`
#'   data set minus p = 2 terminal rows
#'   and `X` is a matrix of ones for the first column
#'   and lagged values of `Y` for the rest of the columns.
#' @keywords fitAutoReg data
"dat_demo_yx"
