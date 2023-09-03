// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-lasso-rep.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVARLassoRep(int time, int burn_in, const arma::vec& constant,
                           const arma::mat& coef, const arma::mat& chol_cov,
                           int n_lambdas, const std::string& crit, int max_iter,
                           double tol) {
  // Step 1: Determine the number of lags in the VAR model
  int num_lags = coef.n_cols / constant.n_elem;

  // Step 2: Simulate data from VAR model with given parameters
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // Step 3: Convert the data into YX representation with lagged variables
  Rcpp::List yx = YX(data, num_lags);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 4: Exclude the constant column from X for standardization
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);

  // Step 5: Fit an OLS model to estimate the constant vector
  arma::mat ols = FitVAROLS(Y, X);
  arma::vec const_b = ols.col(0);

  // Step 6: Standardize the predictor and response variables
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 7: Generate a sequence of lambda values for LASSO regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 8: Fit VAR LASSO model to the standardized data
  arma::mat coef_std_b = FitVARLassoSearch(YStd, XStd, lambdas, crit, max_iter, tol);

  // Step 9: Rescale the estimated coefficients to their original scale
  arma::mat coef_orig = OrigScale(coef_std_b, Y, X_no_constant);

  // Step 10: Combine the estimated constant and original scale coefficients
  arma::mat coef_b = arma::join_horiz(const_b, coef_orig);

  // Step 11: Vectorize the coefficient matrix for bootstrapping
  return arma::vectorise(coef_b);
}
