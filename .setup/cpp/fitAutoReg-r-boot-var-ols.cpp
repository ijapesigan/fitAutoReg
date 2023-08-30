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
//' @inheritParams PBootVAROLS
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original OLS estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'   - **X**: Numeric matrix.
//'     Original `X`
//'   - **Y**: List of numeric matrices.
//'     Bootstrapped `Y`
//'
//' @examples
//' RBootVAROLS(data = dat_p2, p = 2, B = 10)
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

  // Create a matrix to store bootstrap parameter estimates
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);

  // Create a list of bootstrap Y
  Rcpp::List Y_b_list(B);

  for (int b = 0; b < B; ++b) {
    // Residual resampling
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // Simulate new data using bootstrapped residuals
    // and original parameter estimates
    arma::mat Y_b = X * coef.t() + residuals_b;

    // Fit VAR model using bootstrapped data
    arma::mat coef_ols_b = FitVAROLS(Y_b, X);
    arma::vec coef_b = arma::vectorise(coef_ols_b);

    // Store the bootstrapped parameter estimates in the list
    coef_b_mat.col(b) = coef_b;

    // Store the bootstrapped Y in the list
    Y_b_list[b] = Rcpp::wrap(Y_b);
  }

  // Create a list to store the results
  Rcpp::List result;

  // Add coef as the first element
  result["est"] = coef;

  // Store bootstrap coefficients
  result["boot"] = coef_b_mat.t();

  // Store regressors
  result["X"] = X;

  // Store bootstrap Y
  result["Y"] = Y_b_list;

  return result;
}
