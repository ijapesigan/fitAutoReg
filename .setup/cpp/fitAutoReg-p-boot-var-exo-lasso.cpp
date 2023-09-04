// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-exo-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Parametric Bootstrap for the Vector Autoregressive Model
//' with Exogenous Variables
//' Using Lasso Regularization
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @inheritParams PBootVARLasso
//' @inheritParams PBootVARExoOLS
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original OLS estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'
//' @examples
//' data <- dat_p2_exo$data
//' exo_mat <- dat_p2_exo$exo_mat
//' pb <- PBootVARExoLasso(
//'   data = data,
//'   exo_mat = exo_mat,
//'   p = 2,
//'   B = 5,
//'   burn_in = 0,
//'   n_lambdas = 10,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVARExoLasso(const arma::mat& data, const arma::mat& exo_mat,
                            int p, int B, int burn_in, int n_lambdas,
                            const std::string& crit, int max_iter, double tol) {
  // Step 1: Get the number of time points and outcome variables in the data
  // Number of time steps (rows) in 'data'
  int time = data.n_rows;
  // Number of outcome variables (columns) in 'data'
  int num_outcome_vars = data.n_cols;

  // Step 2: Obtain the YX representation of the data
  Rcpp::List yx = YXExo(data, p, exo_mat);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 3: Fit the VAR model using OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Step 4: Extract constant vector and coefficient matrix
  arma::vec const_vec = coef.col(0);
  arma::mat coef_mat = coef.cols(1, coef.n_cols - 1);
  arma::mat coef_lag = coef_mat.cols(0, p * num_outcome_vars - 1);
  arma::mat coef_exo = coef_mat.cols(p * num_outcome_vars, coef_mat.n_cols - 1);

  // Step 5: Calculate residuals and their covariance
  arma::mat residuals = Y - X * coef.t();
  arma::mat cov_residuals = arma::cov(residuals);

  // Step 6: Perform Cholesky decomposition of the covariance matrix
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Step 7: Simulate bootstrapped VAR coefficients using PBootVAROLSSim
  arma::mat sim =
      PBootVARExoLassoSim(B, time, burn_in, const_vec, coef_lag, chol_cov,
                          exo_mat, coef_exo, n_lambdas, crit, max_iter, tol);

  // Step 8: Create a result list containing estimated coefficients
  //         and bootstrapped samples
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = sim;

  // Step 9: Return the result list
  return result;
}
