% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fitAutoReg-model-var-p2-dynr.R
\name{ModelVARP2Dynr}
\alias{ModelVARP2Dynr}
\title{Specify Vector Autoregressive (VAR(p = 2)) Model using dynr}
\usage{
ModelVARP2Dynr(data, outfile = tempfile())
}
\arguments{
\item{data}{Numeric matrix.
The time series data with dimensions \code{t} by \code{k},
where \code{t} is the number of observations
and \code{k} is the number of variables.}

\item{outfile}{a character string of the name of the output C script of model functions to be compiled 
for parameter estimation. The default is the name for a potential temporary file returned by tempfile().}
}
\description{
This function specifies a VAR(p = 2) model
using the \code{\link[dynr:dynr.model]{dynr::dynr.model()}} package.
}
\examples{
ModelVARP2Dynr(data = dat_p2)

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
\code{\link{OrigScale}()},
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
\keyword{fit}
\keyword{fitAutoReg}
