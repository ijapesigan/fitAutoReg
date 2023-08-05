// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Parametric Bootstrap for the Vector Autoregressive Model
//' Using Ordinary Least Squares
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results
//'   in the simulation step.
//'
//' @return List containing the estimates (`est`)
//' and bootstrap estimates (`boot`).
//'
//' @examples
//' pb <- PBootVAROLS(data = dat_demo, p = 2, B = 10, burn_in = 20)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B, int burn_in) {
  // Indices
  int t = data.n_rows;  // Number of observations

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Set parameters
  arma::vec const_vec = coef.col(0);
  arma::mat coef_mat = coef.cols(1, coef.n_cols - 1);

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();

  // Calculate the covariance of residuals
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Result matrix
  arma::mat sim = PBootVAROLSSim(B, t, burn_in, const_vec, coef_mat, chol_cov);

  // Create a list to store the results
  Rcpp::List result;

  // Add coef as the first element
  result["est"] = coef;

  // Add sim as the second element
  result["boot"] = sim;

  // Return the list
  return result;
}
