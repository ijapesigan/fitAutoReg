% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{OrigScale}
\alias{OrigScale}
\title{Return Standardized Estimates to the Original Scale}
\usage{
OrigScale(coef_std, Y, X)
}
\arguments{
\item{coef_std}{Numeric matrix.
Standardized estimates of the autoregression
and cross regression coefficients.}

\item{Y}{Numeric matrix.
Matrix of dependent variables (Y).}

\item{X}{Numeric matrix.
Matrix of predictors (X).}
}
\description{
Return Standardized Estimates to the Original Scale
}
\examples{
Y <- dat_p2_yx$Y
X <- dat_p2_yx$X[, -1] # remove the constant column
YStd <- StdMat(Y)
XStd <- StdMat(X)
coef_std <- FitVAROLS(Y = YStd, X = XStd)
FitVAROLS(Y = Y, X = X)
OrigScale(coef_std = coef_std, Y = Y, X = X)

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
\code{\link{PBootVARExoLasso}()},
\code{\link{PBootVARExoOLS}()},
\code{\link{PBootVARLasso}()},
\code{\link{PBootVAROLS}()},
\code{\link{RBootVARExoLasso}()},
\code{\link{RBootVARExoOLS}()},
\code{\link{RBootVARLasso}()},
\code{\link{RBootVAROLS}()},
\code{\link{SearchVARLasso}()},
\code{\link{StdMat}()}
}
\author{
Ivan Jacob Agaloos Pesigan
}
\concept{Fitting Autoregressive Model Functions}
\keyword{fitAutoReg}
\keyword{utils}
