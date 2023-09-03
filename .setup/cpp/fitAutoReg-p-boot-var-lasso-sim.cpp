// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-lasso-sim.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Function to generate VAR time series data and fit VAR model B times
arma::mat PBootVARLassoSim(int B, int time, int burn_in,
                           const arma::vec& constant, const arma::mat& coef,
                           const arma::mat& chol_cov, int n_lambdas,
                           const std::string& crit, int max_iter, double tol) {
  // Step 1: Determine the number of coefficients (constant + VAR coefficients)
  int num_coef = constant.n_elem + coef.n_elem;

  // Step 2: Initialize a matrix to store bootstrap results
  arma::mat result(B, num_coef, arma::fill::zeros);

  // Step 3: Bootstrap the VAR LASSO coefficients B times
  for (int b = 0; b < B; b++) {
    // Step 4: Call PBootVARLassoRep to perform bootstrap replication
    arma::vec coef_b = PBootVARLassoRep(time, burn_in, constant, coef, chol_cov,
                                        n_lambdas, crit, max_iter, tol);

    // Step 5: Store the bootstrap results in the result matrix
    result.row(b) = arma::trans(coef_b);
  }

  // Step 6: Return the matrix containing bootstrap results
  return result;
}
