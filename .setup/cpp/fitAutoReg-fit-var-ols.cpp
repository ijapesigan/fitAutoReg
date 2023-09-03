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
  // Step 1: Initialize matrices to store QR decomposition results
  arma::mat Q, R;

  // Step 2: Perform QR decomposition of the design matrix X
  arma::qr_econ(Q, R, X);

  // Step 3: Solve the linear system to obtain the coefficient matrix
  //   - Transpose of Q (Q.t()) is multiplied by Y to project Y
  //     onto the column space of X
  //   - R is the upper triangular matrix from the QR decomposition
  //   - arma::solve(R, ...) solves the linear system
  //     R * coef = Q.t() * Y for coef
  arma::mat coef = arma::solve(R, Q.t() * Y);

  // Step 4: Transpose the coefficient matrix to match the desired output format
  //         (columns represent variables, rows represent lags)
  return coef.t();
}
