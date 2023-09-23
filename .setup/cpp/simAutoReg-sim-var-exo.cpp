// -----------------------------------------------------------------------------
// edit .setup/cpp/simAutoReg-sim-var-exo.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive (VAR) Model with Exogenous
//' Variables
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model with exogenous variables.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param time Integer.
//'   Number of time points to simulate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results.
//' @param constant Numeric vector.
//'   The constant term vector of length `k`,
//'   where `k` is the number of variables.
//' @param coef Numeric matrix.
//'   Coefficient matrix with dimensions `k` by `(k * p)`.
//'   Each `k` by `k` block corresponds to the coefficient matrix
//'   for a particular lag.
//' @param chol_cov Numeric matrix.
//'   The Cholesky decomposition of the covariance matrix
//'   of the multivariate normal noise.
//'   It should have dimensions `k` by `k`.
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous covariates with dimensions `time + burn_in` by `x`.
//'   Each column corresponds to a different exogenous variable.
//' @param exo_coef Numeric vector.
//'   Coefficient matrix with dimensions `k` by `x`
//'   associated with the exogenous covariates.
//'
//' @examples
//' set.seed(42)
//' time <- 1000L
//' burn_in <- 200
//' k <- 3
//' p <- 2
//' constant <- c(1, 1, 1)
//' coef <- matrix(
//'   data = c(
//'     0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
//'     0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
//'     0.0, 0.0, 0.6, 0.0, 0.0, 0.3
//'   ),
//'   nrow = k,
//'   byrow = TRUE
//' )
//' chol_cov <- chol(diag(3))
//' exo_mat <- MASS::mvrnorm(
//'   n = time + burn_in,
//'   mu = c(0, 0, 0),
//'   Sigma = diag(3)
//' )
//' exo_coef <- matrix(
//'   data = c(
//'     0.5, 0.0, 0.0,
//'     0.0, 0.5, 0.0,
//'     0.0, 0.0, 0.5
//'   ),
//'   nrow = 3
//' )
//' y <- SimVARExo(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov,
//'   exo_mat = exo_mat,
//'   exo_coef = exo_coef
//' )
//' str(y)
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov,
                    const arma::mat& exo_mat, const arma::mat& exo_coef) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values
  //         for each outcome variable
  data.each_col() = constant;

  // Step 4: Transpose the exogenous matrix for efficient column access
  arma::mat exo_mat_t = exo_mat.t();

  // Step 5: Simulate VAR-Exo data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 5.1: Generate random noise vector
    arma::vec noise = chol_cov * arma::randn(num_outcome_vars);

    // Step 5.2: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 5.3: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 5.4: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 5.5: Iterate over exogenous variables
      for (arma::uword x = 0; x < exo_mat_t.n_rows; x++) {
        // Update data with exogenous variables and their coefficients
        data(j, t) += exo_mat_t(x, t) * exo_coef(j, x);
      }

      // Step 5.6: Add the corresponding element from the noise vector
      data(j, t) += noise(j);
    }
  }

  // Step 6: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 7: Return the transposed data matrix
  return data.t();
}
