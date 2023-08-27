#' Data from the Vector Autoregressive Model (Y) and Lagged Predictors (X)
#' (p = 1)
#'
#' @format A list with elements Y and X where Y is equal to the `dat_p1`
#'   data set minus p = 1 terminal rows
#'   and `X` is a matrix of ones for the first column
#'   and lagged values of `Y` for the rest of the columns.
#' @keywords fitAutoReg data
"dat_p1_yx"
