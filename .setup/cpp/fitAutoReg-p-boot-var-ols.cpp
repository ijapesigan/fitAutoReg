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
//' @inheritParams RBootVAROLS
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results
//'   in the simulation step.
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original OLS estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'
//' @examples
//' pb <- PBootVAROLS(data = dat_p2, p = 2, B = 5, burn_in = 20)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B, int burn_in) {
  // Step 1: Get the number of time points in the data
  int time = data.n_rows;

  // Step 2: Obtain the YX representation of the data
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 3: Fit the VAR model using OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Step 4: Extract constant vector and coefficient matrix
  arma::vec const_vec = coef.col(0);
  arma::mat coef_mat = coef.cols(1, coef.n_cols - 1);

  // Step 5: Calculate residuals and their covariance
  arma::mat residuals = Y - X * coef.t();
  arma::mat cov_residuals = arma::cov(residuals);

  // Step 6: Perform Cholesky decomposition of the covariance matrix
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Step 7: Simulate bootstrapped VAR coefficients using PBootVAROLSSim
  arma::mat sim =
      PBootVAROLSSim(B, time, burn_in, const_vec, coef_mat, chol_cov);

  // Step 8: Create a result list containing estimated coefficients and
  // bootstrapped samples
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = sim;

  // Step 9: Return the result list
  return result;
}
