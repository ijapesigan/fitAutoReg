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
//' @inheritParams PBootVARLasso
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original Lasso estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'   - **X**: Numeric matrix.
//'     Original `X`
//'   - **Y**: List of numeric matrices.
//'     Bootstrapped `Y`
//'
//' @examples
//' RBootVARLasso(
//'   data = dat_p2,
//'   p = 2,
//'   B = 10,
//'   n_lambdas = 100,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B, int n_lambdas,
                         const std::string& crit, int max_iter, double tol) {
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
  arma::mat XStd = StdMat(X_removed);
  arma::mat YStd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Lasso
  arma::mat coef_std =
      FitVARLassoSearch(YStd, XStd, lambdas, "ebic", 1000, 1e-5);
  arma::vec const_vec = ols.col(0);  // OLS constant vector
  arma::mat coef_mat = OrigScale(coef_std, Y, X_removed);  // Lasso coefficients
  arma::mat coef =
      arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();

  // Create a matrix to store bootstrap parameter estimates
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);

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
    arma::mat YStd_b = StdMat(Y);
    arma::mat coef_std_b =
        FitVARLassoSearch(YStd_b, XStd, lambdas, "ebic", 1000, 1e-5);

    // Original scale
    arma::vec const_vec_b = ols_b.col(0);
    arma::mat coef_mat_b = OrigScale(coef_std_b, Y_b, X_removed);
    arma::mat coef_lasso_b = arma::join_horiz(const_vec_b, coef_mat_b);

    arma::vec coef_b = arma::vectorise(coef_lasso_b);

    // Store the bootstrapped parameter estimates in the list
    coef_b_mat.col(b) = coef_b;

    // Store the bootstrapped Y in the list
    Y_list[b] = Rcpp::wrap(Y_b);
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
  result["Y"] = Y_list;

  return result;
}
