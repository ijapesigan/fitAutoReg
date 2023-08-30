// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters
//' using Ordinary Least Squares (OLS)
//'
//' This function estimates the parameters of a VAR model
//' using the Ordinary Least Squares (OLS) method.
//' The OLS method is used to estimate the autoregressive
//' and cross-regression coefficients.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//'
//' @return Matrix of estimated autoregressive
//'   and cross-regression coefficients.
//'
//' @examples
//' Y <- dat_p2_yx$Y
//' X <- dat_p2_yx$X
//' FitVAROLS(Y = Y, X = X)
//'
//' @details
//' The [fitAutoReg::FitVAROLS()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Ordinary Least Squares (OLS) method.
//' Given the input matrices `Y` and `X`,
//' where `Y` is the matrix of dependent variables,
//' and `X` is the matrix of predictors,
//' the function computes the autoregressive
//' and cross-regression coefficients of the VAR model.
//' Note that if the first column of `X` is a vector of ones,
//' the constant vector is also estimated.
//'
//' The steps involved in estimating the VAR model parameters
//' using OLS are as follows:
//'
//' - Compute the QR decomposition of the lagged predictor matrix `X`
//'   using the `qr_econ` function from the Armadillo library.
//' - Extract the `Q` and `R` matrices from the QR decomposition.
//' - Solve the linear system `R * coef = Q.t() * Y`
//'   to estimate the VAR model coefficients `coef`.
//' - The function returns a matrix containing the estimated
//'   autoregressive and cross-regression coefficients of the VAR model.
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVAROLS(const arma::mat& Y, const arma::mat& X) {
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr_econ(Q, R, X);

  // Solve the linear system R * coef = Q.t() * Y
  arma::mat coef = arma::solve(R, Q.t() * Y);

  return coef.t();
}
