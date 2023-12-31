% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fitAutoReg-dat-ml-p1.R
\docType{data}
\name{dat_ml_p1}
\alias{dat_ml_p1}
\title{Data from the Multilevel Vector Autoregressive Model (p = 1)}
\format{
A list of length n = 100 consisting of matrices with
1000 rows (time points) and k = 3 columns (variables)
generated from the p = 1 multilevel vector autoregressive model given by
\deqn{
    Y_{1_{t}}
    =
    1 +
    \mathcal{N} \left( \mu = 0.4, \sigma^2 = 0.01 \right) Y_{1_{t - 1}} +
    0.0 Y_{2_{t - 1}} + 0.0 Y_{3_{t - 1}} +
   \varepsilon_{1_{t}} ,
  }
\deqn{
    Y_{2_{t}}
    =
    1 +
    0.0 Y_{1_{t - 1}} +
    \mathcal{N} \left( \mu = 0.5, \sigma^2 = 0.01 \right)Y_{2_{t - 1}} +
    0.0 Y_{3_{t - 1}} +
    \varepsilon_{2_{t}} ,
  } and
\deqn{
    Y_{3_{t}}
    =
    1 +
    0.0 Y_{1_{t - 1}} + 0.0 Y_{2_{t - 1}} +
    \mathcal{N} \left(  \mu = 0.6, \sigma^2 = 0.01 \right) Y_{3_{t - 1}} +
    \varepsilon_{3_{t}}
  }
which simplifies to
\deqn{
    Y_{1_{t}} = 1 +
    \mathcal{N} \left(  \mu = 0.4, \sigma^2 = 0.01 \right) Y_{1_{t - 1}} +
    \varepsilon_{1_{t}} ,
  }
\deqn{
    Y_{2_{t}} = 1 +
    \mathcal{N} \left(  \mu = 0.5, \sigma^2 = 0.01 \right) Y_{2_{t - 1}} +
    \varepsilon_{2_{t}} ,
  } and
\deqn{
    Y_{3_{t}} = 1 +
    \mathcal{N} \left(  \mu = 0.6, \sigma^2 = 0.01 \right) Y_{3_{t - 1}} +
    \varepsilon_{3_{t}} .
  }
The covariance matrix of process noise is an identity matrix.
}
\usage{
dat_ml_p1
}
\description{
Data from the Multilevel Vector Autoregressive Model (p = 1)
}
\keyword{data}
\keyword{fitAutoReg}
