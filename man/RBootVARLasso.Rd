% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{RBootVARLasso}
\alias{RBootVARLasso}
\title{Residual Bootstrap for the Vector Autoregressive Model
Using Lasso Regularization}
\usage{
RBootVARLasso(data, p, B, n_lambdas, crit, max_iter, tol)
}
\arguments{
\item{data}{Numeric matrix.
The time series data with dimensions \code{t} by \code{k},
where \code{t} is the number of observations
and \code{k} is the number of variables.}

\item{p}{Integer.
The order of the VAR model (number of lags).}

\item{B}{Integer.
Number of bootstrap samples to generate.}

\item{n_lambdas}{Integer.
Number of lambdas to generate.}

\item{crit}{Character string.
Information criteria to use.
Valid values include \code{"aic"}, \code{"bic"}, and \code{"ebic"}.}

\item{max_iter}{Integer.
The maximum number of iterations
for the coordinate descent algorithm
(e.g., \code{max_iter = 10000}).}

\item{tol}{Numeric.
Convergence tolerance.
The algorithm stops when the change in coefficients
between iterations is below this tolerance
(e.g., \code{tol = 1e-5}).}
}
\value{
List with the following elements:
\itemize{
\item \strong{est}: Numeric matrix.
Original Lasso estimate of the coefficient matrix.
\item \strong{boot}: Numeric matrix.
Matrix of vectorized bootstrap estimates of the coefficient matrix.
\item \strong{X}: Numeric matrix.
Original \code{X}
\item \strong{Y}: List of numeric matrices.
Bootstrapped \code{Y}
}
}
\description{
Residual Bootstrap for the Vector Autoregressive Model
Using Lasso Regularization
}
\examples{
rb <- RBootVARLasso(
  data = dat_p2,
  p = 2,
  B = 5,
  n_lambdas = 10,
  crit = "ebic",
  max_iter = 1000,
  tol = 1e-5
)
str(rb)

}
\seealso{
Other Fitting Autoregressive Model Functions: 
\code{\link{FitMLVARDynr}()},
\code{\link{FitMLVARMplus}()},
\code{\link{FitVARDynr}()},
\code{\link{FitVARLassoSearch}()},
\code{\link{FitVARLasso}()},
\code{\link{FitVARMplus}()},
\code{\link{FitVAROLS}()},
\code{\link{LambdaSeq}()},
\code{\link{ModelVARP1Dynr}()},
\code{\link{ModelVARP2Dynr}()},
\code{\link{OrigScale}()},
\code{\link{PBootVARExoLasso}()},
\code{\link{PBootVARExoOLS}()},
\code{\link{PBootVARLasso}()},
\code{\link{PBootVAROLS}()},
\code{\link{RBootVARExoLasso}()},
\code{\link{RBootVARExoOLS}()},
\code{\link{RBootVAROLS}()},
\code{\link{SearchVARLasso}()},
\code{\link{StdMat}()}
}
\author{
Ivan Jacob Agaloos Pesigan
}
\concept{Fitting Autoregressive Model Functions}
\keyword{fitAutoReg}
\keyword{rb}
