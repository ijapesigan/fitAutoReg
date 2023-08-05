// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols-rep.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant,
                         const arma::mat& coef, const arma::mat& chol_cov) {
  // Indices
  int k = constant.n_elem;  // Number of variables
  int q = coef.n_cols;      // Dimension of the coefficient matrix
  int p = q / k;            // Order of the VAR model (number of lags)

  // Simulate data
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // OLS
  arma::mat pb_coef = FitVAROLS(Y, X);

  return arma::vectorise(pb_coef);
}
