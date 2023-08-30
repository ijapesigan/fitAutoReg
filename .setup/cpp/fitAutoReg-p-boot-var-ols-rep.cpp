// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols-rep.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant,
                         const arma::mat& coef, const arma::mat& chol_cov) {
  // Order of the VAR model (number of lags)
  int p = coef.n_cols / constant.n_elem;

  // Simulate data
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // OLS
  arma::mat coef_b = FitVAROLS(Y, X);

  return arma::vectorise(coef_b);
}
