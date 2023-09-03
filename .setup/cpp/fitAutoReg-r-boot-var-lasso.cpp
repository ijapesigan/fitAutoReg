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
//' @inheritParams RBootVAROLS
//' @inheritParams FitVARLassoSearch
//' @inheritParams LambdaSeq
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
//' rb <- RBootVARLasso(
//'   data = dat_p2,
//'   p = 2,
//'   B = 5,
//'   n_lambdas = 10,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B, int n_lambdas, const std::string& crit, int max_iter, double tol) {
  // Step 1: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);
  int time = Y.n_rows;

  // Step 2: Fit a VAR model using OLS to obtain the VAR coefficients
  arma::mat ols = FitVAROLS(Y, X);

  // Step 3: Standardize the data
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 4: Generate a sequence of lambda values for Lasso regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 5: Fit VAR Lasso to obtain standardized coefficients
  arma::mat coef_std = FitVARLassoSearch(YStd, XStd, lambdas, "ebic", max_iter, tol);

  // Step 6: Extract the constant vector from OLS results
  arma::vec const_vec = ols.col(0);

  // Step 7: Transform standardized coefficients back to the original scale
  arma::mat coef_mat = OrigScale(coef_std, Y, X_no_constant);

  // Step 8: Combine the constant and coefficient matrices
  arma::mat coef = arma::join_horiz(const_vec, coef_mat);

  // Step 9: Calculate residuals based on the estimated VAR coefficients
  arma::mat residuals = Y - X * coef.t();

  // Step 10: Prepare containers for bootstrap results
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);
  Rcpp::List Y_list(B);

  // Step 11: Perform B bootstrap simulations
  for (int b = 0; b < B; ++b) {
    // 11.1: Randomly select rows from residuals to create a new residuals matrix for the bootstrap sample
    arma::mat residuals_b = residuals.rows(arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // 11.2: Generate a new response matrix Y_b by adding the new residuals to X * coef.t()
    arma::mat Y_b = X * coef.t() + residuals_b;

    // 11.3: Fit a VAR model using OLS to obtain VAR coefficients for the bootstrap sample
    arma::mat ols_b = FitVAROLS(Y_b, X);

    // 11.4: Standardize the Y_b matrix
    arma::mat YStd_b = StdMat(Y);

    // 11.5: Fit VAR Lasso to obtain standardized coefficients for the bootstrap sample
    arma::mat coef_std_b = FitVARLassoSearch(YStd_b, XStd, lambdas, "ebic", max_iter, tol);

    // 11.6: Extract the constant vector from OLS results for the bootstrap sample
    arma::vec const_vec_b = ols_b.col(0);

    // 11.7: Transform standardized coefficients back to the original scale for the bootstrap sample
    arma::mat coef_mat_b = OrigScale(coef_std_b, Y_b, X_no_constant);

    // 11.8: Combine the constant and coefficient matrices for the bootstrap sample
    arma::mat coef_lasso_b = arma::join_horiz(const_vec_b, coef_mat_b);

    // 11.9: Vectorize the coefficients and store them in coef_b_mat
    arma::vec coef_b = arma::vectorise(coef_lasso_b);
    coef_b_mat.col(b) = coef_b;

    // 11.10: Store the Y_b matrix as an Rcpp list element
    Y_list[b] = Rcpp::wrap(Y_b);
  }

  // Step 12: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = coef_b_mat.t();
  result["X"] = X;
  result["Y"] = Y_list;

  // Step 13: Return the list of results
  return result;
}
