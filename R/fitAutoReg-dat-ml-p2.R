#' Data from the Multilevel Vector Autoregressive Model (p = 2)
#'
#' @format A list of length n = 100 consisting of matrices with
#'   1000 rows (time points) and k = 3 columns (variables)
#'   generated from the p = 2 multilevel vector autoregressive model given by
#'   \deqn{
#'     Y_{1_{t}}
#'     =
#'     1 +
#'     \mathcal{N} \left( \mu = 0.4, \sigma^2 = 0.01 \right) Y_{1_{t - 1}} +
#'     0.0 Y_{2_{t - 1}} + 0.0 Y_{3_{t - 1}} +
#'     \mathcal{N} \left( \mu = 0.1, \sigma^2 = 0.01 \right) Y_{1_{t - 2}} +
#'     0.0 Y_{2_{t - 2}} + 0.0 Y_{3_{t - 2}} +
#'    \varepsilon_{1_{t}} ,
#'   }
#'   \deqn{
#'     Y_{2_{t}}
#'     =
#'     1 +
#'     0.0 Y_{1_{t - 1}} +
#'     \mathcal{N} \left( \mu = 0.5, \sigma^2 = 0.01 \right) Y_{2_{t - 1}} +
#'     0.0 Y_{3_{t - 1}} +
#'     0.0 Y_{1_{t - 2}} +
#'     \mathcal{N} \left( \mu = 0.2, \sigma^2 = 0.01 \right) Y_{2_{t - 2}} +
#'     0.0 Y_{3_{t - 2}} +
#'     \varepsilon_{2_{t}} ,
#'   } and
#'   \deqn{
#'     Y_{3_{t}}
#'     =
#'     1 +
#'     0.0 Y_{1_{t - 1}} + 0.0 Y_{2_{t - 1}} +
#'     \mathcal{N} \left( \mu = 0.6, \sigma^2 = 0.01 \right) Y_{3_{t - 1}} +
#'     0.0 Y_{1_{t - 2}} + 0.0 Y_{2_{t - 2}} +
#'     \mathcal{N} \left( \mu = 0.3, \sigma^2 = 0.01 \right) Y_{3_{t - 2}} +
#'     \varepsilon_{3_{t}}
#'   }
#'   which simplifies to
#'   \deqn{
#'     Y_{1_{t}} = 1 +
#'     \mathcal{N} \left( \mu = 0.4, \sigma^2 = 0.01 \right) Y_{1_{t - 1}} +
#'     \mathcal{N} \left( \mu = 0.1, \sigma^2 = 0.01 \right) Y_{1_{t - 2}} +
#'     \varepsilon_{1_{t}} ,
#'   }
#'   \deqn{
#'     Y_{2_{t}} = 1 +
#'     \mathcal{N} \left( \mu = 0.5, \sigma^2 = 0.01 \right) Y_{2_{t - 1}} +
#'     \mathcal{N} \left( \mu = 0.2, \sigma^2 = 0.01 \right) Y_{2_{t - 2}} +
#'     \varepsilon_{2_{t}} ,
#'   } and
#'   \deqn{
#'     Y_{3_{t}} = 1 +
#'     \mathcal{N} \left( \mu = 0.6, \sigma^2 = 0.01 \right) Y_{3_{t - 1}} +
#'     \mathcal{N} \left( \mu = 0.3, \sigma^2 = 0.01 \right) Y_{3_{t - 2}} +
#'     \varepsilon_{3_{t}} .
#'   }
#'   The covariance matrix of process noise is an identity matrix.
#' @keywords fitAutoReg data
"dat_ml_p2"
