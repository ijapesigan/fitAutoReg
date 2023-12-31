% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fitAutoReg-dat-p2.R
\docType{data}
\name{dat_p2}
\alias{dat_p2}
\title{Data from the Vector Autoregressive Model (p = 2)}
\format{
A matrix with 1000 rows (time points) and k = 3 columns (variables)
generated from the p = 2 vector autoregressive model given by
\deqn{
    Y_{1_{t}}
    =
    1 +
    0.4 Y_{1_{t - 1}} + 0.0 Y_{2_{t - 1}} + 0.0 Y_{3_{t - 1}} +
    0.1 Y_{1_{t - 2}} + 0.0 Y_{2_{t - 2}} + 0.0 Y_{3_{t - 2}} +
   \varepsilon_{1_{t}} ,
  }
\deqn{
    Y_{2_{t}}
    =
    1 +
    0.0 Y_{1_{t - 1}} + 0.5 Y_{2_{t - 1}} + 0.0 Y_{3_{t - 1}} +
    0.0 Y_{1_{t - 2}} + 0.2 Y_{2_{t - 2}} + 0.0 Y_{3_{t - 2}} +
    \varepsilon_{2_{t}} ,
  } and
\deqn{
    Y_{3_{t}}
    =
    1 +
    0.0 Y_{1_{t - 1}} + 0.0 Y_{2_{t - 1}} + 0.6 Y_{3_{t - 1}} +
    0.0 Y_{1_{t - 2}} + 0.0 Y_{2_{t - 2}} + 0.3 Y_{3_{t - 2}} +
    \varepsilon_{3_{t}}
  }
which simplifies to
\deqn{
    Y_{1_{t}} = 1 + 0.4 Y_{1_{t - 1}} + 0.1 Y_{1_{t - 2}} +
    \varepsilon_{1_{t}} ,
  }
\deqn{
    Y_{2_{t}} = 1 + 0.5 Y_{2_{t - 1}} + 0.2 Y_{2_{t - 2}} +
    \varepsilon_{2_{t}} ,
  } and
\deqn{
    Y_{3_{t}} = 1 + 0.6 Y_{3_{t - 1}} + 0.3 Y_{3_{t - 2}} +
    \varepsilon_{3_{t}} .
  }
The covariance matrix of process noise is an identity matrix.
}
\usage{
dat_p2
}
\description{
Data from the Vector Autoregressive Model (p = 2)
}
\keyword{data}
\keyword{fitAutoReg}
