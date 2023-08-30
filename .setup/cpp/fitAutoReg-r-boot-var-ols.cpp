// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-r-boot-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Residual Bootstrap for the Vector Autoregressive Model
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
//'
//' @return List with the following elements:
//'   - List of bootstrap estimates
//'   - original `X`
//'   - List of bootstrapped `Y`
//'
//' @examples
//' rb <- RBootVAROLS(data = dat_p2, p = 2, B = 10)
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVAROLS(const arma::mat& data, int p, int B) {
  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Indices
  int time = Y.n_rows;  // Number of observations

  // OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Residuals
  arma::mat residuals = Y - X * coef.t();

  // Create a list to store bootstrap parameter estimates
  Rcpp::List coef_list(B);

  // Create a list of bootstrap Y
  Rcpp::List Y_list(B);

  for (int b = 0; b < B; ++b) {
    // Residual resampling
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // Simulate new data using bootstrapped residuals
    // and original parameter estimates
    arma::mat Y_b = X * coef.t() + residuals_b;

    // Fit VAR model using bootstrapped data
    arma::mat coef_b = FitVAROLS(Y_b, X);

    // Store the bootstrapped parameter estimates in the list
    coef_list[b] = Rcpp::wrap(coef_b);

    // Store the bootstrapped Y in the list
    Y_list[b] = Rcpp::wrap(Y_b);
  }

  // Create a list to store the results
  Rcpp::List result;

  // Store bootstrap coefficients
  result["coef"] = coef_list;

  // Store regressors
  result["X"] = X;

  // Store bootstrap Y
  result["Y"] = Y_list;

  return result;
}
