% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{SearchVARLasso}
\alias{SearchVARLasso}
\title{Compute AIC, BIC, and EBIC for Lasso Regularization}
\usage{
SearchVARLasso(YStd, XStd, lambdas, max_iter, tol)
}
\arguments{
\item{YStd}{Numeric matrix.
Matrix of standardized dependent variables (Y).}

\item{XStd}{Numeric matrix.
Matrix of standardized predictors (X).
\code{XStd} should not include a vector of ones in column one.}

\item{lambdas}{Numeric vector.
Lasso hyperparameter.
The regularization strength controlling the sparsity.}

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
\item \strong{criteria}: Matrix with columns for
lambda, AIC, BIC, and EBIC values.
\item \strong{fit}: List of matrices containing
the estimated autoregressive
and cross-regression coefficients for each lambda.
}
}
\description{
This function computes the Akaike Information Criterion (AIC),
Bayesian Information Criterion (BIC),
and Extended Bayesian Information Criterion (EBIC)
for a given matrix of predictors \code{X}, a matrix of outcomes \code{Y},
and a vector of lambda hyperparameters for Lasso regularization.
}
\examples{
YStd <- StdMat(dat_p2_yx$Y)
XStd <- StdMat(dat_p2_yx$X[, -1])
lambdas <- 10^seq(-5, 5, length.out = 100)
search <- SearchVARLasso(YStd = YStd, XStd = XStd, lambdas = lambdas,
  max_iter = 10000, tol = 1e-5)
plot(x = 1:nrow(search$criteria), y = search$criteria[, 4],
  type = "b", xlab = "lambda", ylab = "EBIC")

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
\code{\link{RBootVARLasso}()},
\code{\link{RBootVAROLS}()},
\code{\link{StdMat}()}
}
\author{
Ivan Jacob Agaloos Pesigan
}
\concept{Fitting Autoregressive Model Functions}
\keyword{fit}
\keyword{fitAutoReg}
