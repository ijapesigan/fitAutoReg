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
//' PBootVARLasso(
//'   data = dat_p2,
//'   p = 2,
//'   B = 10,
//'   burn_in = 20,
//'   n_lambdas = 100,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVARLasso(const arma::mat& data, int p, int B, int burn_in,
                         int n_lambdas, const std::string& crit, int max_iter,
                         double tol) {
  // Number of observations
  int t = data.n_rows;

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_removed = X.cols(1, X.n_cols - 1);

  // OLS
  arma::mat ols = FitVAROLS(Y, X);

  // Standardize
  arma::mat XStd = StdMat(X_removed);
  arma::mat YStd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Lasso
  arma::mat pb_std = FitVARLassoSearch(YStd, XStd, lambdas, "ebic", 1000, 1e-5);

  // Set parameters
  arma::vec const_vec = ols.col(0);                      // OLS constant vector
  arma::mat coef_mat = OrigScale(pb_std, Y, X_removed);  // Lasso coefficients
  arma::mat coef =
      arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();
  // arma::mat residuals_tmp = Y.each_row() - const_vec.t();
  // arma::mat residuals = residuals_tmp - X_removed * coef_mat.t();

  // Calculate the covariance of residuals
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Result matrix
  arma::mat sim = PBootVARLassoSim(B, t, burn_in, const_vec, coef_mat, chol_cov,
                                   n_lambdas, crit, max_iter, tol);

  // Create a list to store the results
  Rcpp::List result;

  // Add coef as the first element
  result["est"] = coef;

  // Add sim as the second element
  result["boot"] = sim;

  // Return the list
  return result;
}
