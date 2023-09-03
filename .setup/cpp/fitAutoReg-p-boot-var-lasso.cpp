// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Parametric Bootstrap for the Vector Autoregressive Model
//' Using Lasso Regularization
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @inheritParams PBootVAROLS
//' @inheritParams LambdaSeq
//' @inheritParams FitVARLassoSearch
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original Lasso estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'
//' @examples
//' pb <- PBootVARLasso(
//'   data = dat_p2,
//'   p = 2,
//'   B = 5,
//'   burn_in = 20,
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
Rcpp::List PBootVARLasso(const arma::mat& data, int p, int B, int burn_in,
                         int n_lambdas, const std::string& crit, int max_iter,
                         double tol) {
  // Step 1: Get the number of time periods
  int time = data.n_rows;

  // Step 2: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  
  // Step 3: Remove the constant term from the lagged variables
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);

  // Step 4: Fit a VAR model using OLS to obtain the constant term and VAR coefficients
  arma::mat ols = FitVAROLS(Y, X);

  // Step 5: Standardize the predictor and response variables
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 6: Generate a sequence of lambda values for LASSO regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 7: Fit VAR LASSO using the "ebic" criterion
  arma::mat pb_std = FitVARLassoSearch(YStd, XStd, lambdas, "ebic", 1000, 1e-5);

  // Step 8: Extract the constant term from the OLS results
  arma::vec const_vec = ols.col(0);

  // Step 9: Rescale the VAR LASSO coefficients to the original scale
  arma::mat coef_mat = OrigScale(pb_std, Y, X_no_constant);

  // Step 10: Combine the constant and VAR coefficients
  arma::mat coef = arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Step 11: Calculate residuals, their covariance, and the Cholesky decomposition of the covariance
  arma::mat residuals = Y - X * coef.t();
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Step 12: Perform bootstrap simulations for VAR LASSO
  arma::mat sim = PBootVARLassoSim(B, time, burn_in, const_vec, coef_mat, chol_cov,
                                   n_lambdas, crit, max_iter, tol);

  // Step 13: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = sim;

  // Step 14: Return the list of results
  return result;
}
