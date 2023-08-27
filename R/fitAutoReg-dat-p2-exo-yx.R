#' Data from the Vector Autoregressive Model (Y) and Lagged Predictors
#' and Exogenous Variables (X)  (p = 2)
#'
#' @format A list with elements Y and X where Y
#'   is equal to the k = 3 autoregressive variables
#'   of the `dat_p2_exo`
#'   data set minus p = 2 terminal rows
#'   and `X` is a matrix of ones for the first column,
#'   lagged values of `Y`, and m = 3 exogenous variables.
#' @keywords fitAutoReg data
"dat_p2_exo_yx"
