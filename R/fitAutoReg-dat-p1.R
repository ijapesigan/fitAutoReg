#' Data from the Vector Autoregressive Model (p = 1)
#'
#' @format A matrix with 1000 rows (time points) and k = 3 columns (variables)
#'   generated from the p = 1 vector autoregressive model given by
#'   \deqn{
#'     Y_{1_{t}}
#'     =
#'     1 +
#'     0.4 Y_{1_{t - 1}} + 0.0 Y_{2_{t - 1}} + 0.0 Y_{3_{t - 1}} +
#'    \varepsilon_{1_{t}} ,
#'   }
#'   \deqn{
#'     Y_{2_{t}}
#'     =
#'     1 +
#'     0.0 Y_{1_{t - 1}} + 0.5 Y_{2_{t - 1}} + 0.0 Y_{3_{t - 1}} +
#'     \varepsilon_{2_{t}} ,
#'   } and
#'   \deqn{
#'     Y_{3_{t}}
#'     =
#'     1 +
#'     0.0 Y_{1_{t - 1}} + 0.0 Y_{2_{t - 1}} + 0.6 Y_{3_{t - 1}} +
#'     \varepsilon_{3_{t}}
#'   }
#'   which simplifies to
#'   \deqn{
#'     Y_{1_{t}} = 1 + 0.4 Y_{1_{t - 1}} +
#'     \varepsilon_{1_{t}} ,
#'   }
#'   \deqn{
#'     Y_{2_{t}} = 1 + 0.5 Y_{2_{t - 1}} +
#'     \varepsilon_{2_{t}} ,
#'   } and
#'   \deqn{
#'     Y_{3_{t}} = 1 + 0.6 Y_{3_{t - 1}} +
#'     \varepsilon_{3_{t}} .
#'   }
#'   The covariance matrix of process noise is an identity matrix.
#' @keywords fitAutoReg data
"dat_p1"
