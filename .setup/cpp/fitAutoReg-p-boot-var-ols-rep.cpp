// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols-rep.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant, const arma::mat& coef, const arma::mat& chol_cov) {
  // Step 1: Calculate the number of lags in the VAR model
  int num_lags = coef.n_cols / constant.n_elem;

  // Step 2: Simulate a VAR process using the SimVAR function
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // Step 3: Create lagged matrices X and Y using the YX function
  Rcpp::List yx = YX(data, num_lags);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 4: Estimate VAR coefficients using the FitVAROLS function
  arma::mat coef_b = FitVAROLS(Y, X);

  // Step 5: Return the estimated coefficients as a vector
  return arma::vectorise(coef_b);
}
