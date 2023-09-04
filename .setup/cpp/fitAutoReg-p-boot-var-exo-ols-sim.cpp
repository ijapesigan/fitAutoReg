// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-exo-ols-sim.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Function to generate VAR time series data and fit VAR model B times
arma::mat PBootVARExoOLSSim(int B, int time, int burn_in,
                            const arma::vec& constant, const arma::mat& coef,
                            const arma::mat& chol_cov, const arma::mat& exo_mat,
                            const arma::mat& exo_coef) {
  // Step 1: Calculate the total number of coefficients in the VAR model
  int num_coef = constant.n_elem + coef.n_elem + exo_coef.n_elem;

  // Step 2: Initialize the result matrix
  //         to store bootstrapped coefficient estimates
  arma::mat result(B, num_coef, arma::fill::zeros);

  // Step 3: Perform bootstrapping B times
  for (int b = 0; b < B; b++) {
    // Step 4: Obtain bootstrapped VAR coefficient estimates
    //         using PBootVAROLSRep
    arma::vec coef_est = PBootVARExoOLSRep(time, burn_in, constant, coef,
                                           chol_cov, exo_mat, exo_coef);

    // Step 5: Store the estimated coefficients in the result matrix
    result.row(b) = arma::trans(coef_est);
  }

  // Step 6: Return the result matrix containing
  //         bootstrapped coefficient estimates
  return result;
}
