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
//' @inheritParams YX
//' @param B Integer.
//'   Number of bootstrap samples to generate.
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
//' rb <- RBootVAROLS(data = dat_p2, p = 2, B = 5)
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVAROLS(const arma::mat& data, int p, int B) {
  // Step 1: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  int time = Y.n_rows;

  // Step 2: Fit a VAR model using OLS to obtain the VAR coefficients
  arma::mat coef = FitVAROLS(Y, X);

  // Step 3: Calculate residuals based on the estimated VAR coefficients
  arma::mat residuals = Y - X * coef.t();

  // Step 4: Prepare containers for bootstrap results
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);
  Rcpp::List Y_b_list(B);

  // Step 5: Perform B bootstrap simulations
  for (int b = 0; b < B; ++b) {
    // 5.1: Randomly select rows from residuals to create a new residuals matrix
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // 5.2: Generate a new response matrix Y_b by adding the new residuals to X
    // * coef.t()
    arma::mat Y_b = X * coef.t() + residuals_b;

    // 5.3: Fit a VAR model using OLS to obtain VAR coefficients for the
    // bootstrap sample
    arma::mat coef_ols_b = FitVAROLS(Y_b, X);

    // 5.4: Vectorize the coefficients and store them in coef_b_mat
    arma::vec coef_b = arma::vectorise(coef_ols_b);
    coef_b_mat.col(b) = coef_b;

    // 5.5: Store the Y_b matrix as an Rcpp list element
    Y_b_list[b] = Rcpp::wrap(Y_b);
  }

  // Step 6: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = coef_b_mat.t();
  result["X"] = X;
  result["Y"] = Y_b_list;

  // Step 7: Return the list of results
  return result;
}
