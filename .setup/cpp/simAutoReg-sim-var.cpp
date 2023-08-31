// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive (VAR) Model
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model.
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
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
//'
//' @examples
//' set.seed(42)
//' time <- 50L
//' burn_in <- 10L
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
//' y <- SimVAR(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov
//' )
//' head(y)
//'
//' @details
//' The [SimVAR()] function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model.
//' The VAR model is defined by the constant term `constant`,
//' the coefficient matrix `coef`,
//' and the Cholesky decomposition of the covariance matrix
//' of the multivariate normal process noise `chol_cov`.
//' The generated time series data follows a VAR(p) process,
//' where `p` is the number of lags specified by the size of `coef`.
//' The generated data includes a burn-in period,
//' which is excluded before returning the results.
//'
//' The steps involved in generating the VAR time series data are as follows:
//'
//' - Extract the number of variables `k` and the number of lags `p`
//'   from the input.
//' - Create a matrix `data` of size `k` by (`time + burn_in`)
//'   to store the generated VAR time series data.
//' - Set the initial values of the matrix `data`
//'   using the constant term `constant`.
//' - For each time point starting from the `p`-th time point
//'   to `time + burn_in - 1`:
//'   * Generate a vector of random noise
//'     from a multivariate normal distribution
//'     with mean 0 and covariance matrix `chol_cov`.
//'   * Generate the VAR time series values for each variable `j` at time `t`
//'     using the formula:
//'     \deqn{Y_{tj} = \text{constant}_j +
//'     \sum_{l = 1}^{p} \sum_{m = 1}^{k} (\text{coef}_{jm} * Y_{im}) +
//'     \text{noise}_{j}}
//'     where \eqn{Y_{tj}} is the value of variable `j` at time `t`,
//'     \eqn{constant_j} is the constant term for variable `j`,
//'     \eqn{coef_{jm}} are the coefficients for variable `j`
//'     from lagged variables up to order `p`,
//'     \eqn{Y_{tm}} are the lagged values of variable `m`
//'     up to order `p` at time `t`,
//'     and \eqn{noise_{j}} is the element `j`
//'     from the generated vector of random process noise.
//' - Transpose the matrix `data` and return only
//'   the required time period after the burn-in period,
//'   which is from column `burn_in` to column `time + burn_in - 1`.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  int total_time = burn_in + time;

  // Matrix to store the generated VAR time series data
  arma::mat data(k, total_time);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < total_time; t++) {
    // Generate noise from a multivariate normal distribution
    arma::vec noise = arma::randn(k);
    arma::vec mult_noise = chol_cov * noise;

    // Generate eta_t vector
    for (int j = 0; j < k; j++) {
      for (int lag = 0; lag < p; lag++) {
        for (int l = 0; l < k; l++) {
          data(j, t) += coef(j, lag * k + l) * data(l, t - lag - 1);
        }
      }

      data(j, t) += mult_noise(j);
    }
  }

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}

// Dependencies
