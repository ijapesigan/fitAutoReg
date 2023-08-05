// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols-sim.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Function to generate VAR time series data and fit VAR model B times
arma::mat PBootVAROLSSim(int B, int time, int burn_in,
                         const arma::vec& constant, const arma::mat& coef,
                         const arma::mat& chol_cov) {
  int num_coef = constant.n_elem + coef.n_elem;
  arma::mat result(B, num_coef, arma::fill::zeros);

  for (int i = 0; i < B; i++) {
    arma::vec coef_est =
        PBootVAROLSRep(time, burn_in, constant, coef, chol_cov);
    result.row(i) = arma::trans(coef_est);
  }

  return result;
}
