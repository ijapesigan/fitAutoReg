// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-exo-ols-rep.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVARExoOLSRep(int time, int burn_in, const arma::vec& constant,
                            const arma::mat& coef, const arma::mat& chol_cov,
                            const arma::mat& exo_mat,
                            const arma::mat& exo_coef) {
  // Step 1: Calculate the number of lags in the VAR model
  int num_lags = coef.n_cols / constant.n_elem;

  // Step 2: Simulate a VAR process using the SimVARExo function
  arma::mat data =
      SimVARExo(time, burn_in, constant, coef, chol_cov, exo_mat, exo_coef);

  // Step 3: Create lagged matrices X and Y using the YXExo function
  Rcpp::List yx = YXExo(data, num_lags, exo_mat);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 4: Estimate VAR coefficients using the FitVAROLS function
  arma::mat coef_b = FitVAROLS(Y, X);

  // Step 5: Return the estimated coefficients as a vector
  return arma::vectorise(coef_b);
}
