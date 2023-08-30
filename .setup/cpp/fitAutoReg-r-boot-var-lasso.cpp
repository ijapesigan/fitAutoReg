// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-r-boot-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Residual Bootstrap for the Vector Autoregressive Model
//' Using Lasso Regularization
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
//' @param n_lambdas Integer.
//'   Number of lambdas to generate.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm
//'   (e.g., `max_iter = 10000`).
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance
//'   (e.g., `tol = 1e-5`).
//' @param crit Character string.
//'   Information criteria to use.
//'   Valid values include `"aic"`, `"bic"`, and `"ebic"`.
//'
//' @return List with the following elements:
//'   - List of bootstrap estimates
//'   - original `X`
//'   - List of bootstrapped `Y`
//'
//' @examples
//' pb <- RBootVARLasso(data = dat_p2, p = 2, B = 10,
//'   n_lambdas = 100, crit = "ebic", max_iter = 1000, tol = 1e-5)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B,
                         int n_lambdas, const std::string& crit, int max_iter,
                         double tol) {
  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_removed = X.cols(1, X.n_cols - 1);

  // Indices
  int time = Y.n_rows;  // Number of observations

  // OLS
  arma::mat ols = FitVAROLS(Y, X);

  // Standardize
  arma::mat Xstd = StdMat(X_removed);
  arma::mat Ystd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(Ystd, Xstd, n_lambdas);

  // Lasso
  arma::mat coef_std = FitVARLassoSearch(Ystd, Xstd, lambdas, "ebic", 1000, 1e-5);
  arma::vec const_vec = ols.col(0);                        // OLS constant vector
  arma::mat coef_mat = OrigScale(coef_std, Y, X_removed);  // Lasso coefficients
  arma::mat coef =
      arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Calculate the residuals
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
    arma::mat ols_b = FitVAROLS(Y_b, X);
    arma::mat Ystd_b = StdMat(Y);
    arma::mat coef_std_b = FitVARLassoSearch(Ystd_b, Xstd, lambdas, "ebic", 1000, 1e-5);

    // Original scale
    arma::vec const_vec_b = ols_b.col(0);
    arma::mat coef_mat_b = OrigScale(coef_std_b, Y_b, X_removed);
    arma::mat coef_b =
      arma::join_horiz(const_vec_b, coef_mat_b);

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
