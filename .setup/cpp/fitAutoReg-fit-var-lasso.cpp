// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters using Lasso Regularization
//'
//' This function estimates the parameters of a VAR model
//' using the Lasso regularization method with cyclical coordinate descent.
//' The Lasso method is used to estimate the autoregressive
//' and cross-regression coefficients with sparsity.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Ystd Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param Xstd Numeric matrix.
//'   Matrix of standardized predictors (X).
//'   `Xstd` should not include a vector of ones in column one.
//' @param lambda Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm
//'   (e.g., `max_iter = 10000`).
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance
//'   (e.g., `tol = 1e-5`).
//'
//' @return Matrix of estimated autoregressive and
//' cross-regression coefficients.
//'
//' @examples
//' Ystd <- StdMat(dat_p2_yx$Y)
//' Xstd <- StdMat(dat_p2_yx$X[, -1])
//' lambda <- 73.90722
//' FitVARLasso(Ystd = Ystd, Xstd = Xstd, lambda = lambda,
//'   max_iter = 10000, tol = 1e-5)
//'
//' @details
//' The [fitAutoReg::FitVARLasso()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Lasso regularization method.
//' Given the input matrices `Ystd` and `Xstd`,
//' where `Ystd` is the matrix of standardized dependent variables,
//' and `Xstd` is the matrix of standardized predictors,
//' the function computes the autoregressive and cross-regression coefficients
//' of the VAR model with sparsity induced by the Lasso regularization.
//'
//' The steps involved in estimating the VAR model parameters
//' using Lasso are as follows:
//'
//' - **Initialization**: The function initializes the coefficient matrix
//'   `beta` with OLS estimates.
//'   The `beta` matrix will store the estimated autoregressive and
//'   cross-regression coefficients.
//' - **Coordinate Descent Loop**: The function performs
//'   the cyclical coordinate descent algorithm
//'   to estimate the coefficients iteratively.
//'   The loop iterates `max_iter` times,
//'   or until convergence is achieved.
//'   The outer loop iterates over the predictor variables
//'   (columns of `Xstd`),
//'   while the inner loop iterates over the outcome variables
//'   (columns of `Ystd`).
//' - **Coefficient Update**: For each predictor variable (column of `Xstd`),
//'   the function iteratively updates the corresponding column of `beta`
//'   using the coordinate descent algorithm with L1 norm regularization
//'   (Lasso).
//'   The update involves calculating the soft-thresholded value `c`,
//'   which encourages sparsity in the coefficients.
//'   The algorithm continues until the change in coefficients
//'   between iterations is below the specified tolerance `tol`
//'   or when the maximum number of iterations is reached.
//' - **Convergence Check**: The function checks for convergence
//'   by comparing the current `beta`
//'   matrix with the previous iteration's `beta_old`.
//'   If the maximum absolute difference between `beta` and `beta_old`
//'   is below the tolerance `tol`,
//'   the algorithm is considered converged, and the loop exits.
//'
//' @seealso
//' The [fitAutoReg::FitVAROLS()] function for estimating VAR model parameters
//' using OLS.
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLasso(const arma::mat& Ystd, const arma::mat& Xstd,
                      const double& lambda, int max_iter, double tol) {
  int q = Xstd.n_cols;  // Number of predictors (excluding the intercept column)
  int k = Ystd.n_cols;  // Number of outcomes

  // OLS starting values
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr(Q, R, Xstd);
  // Solve the linear system R * beta = Q.t() * Ystd
  arma::mat beta = arma::solve(R, Q.t() * Ystd);

  // Coordinate Descent Loop
  for (int iter = 0; iter < max_iter; iter++) {
    arma::mat beta_old = beta;  // Initialize beta_old
                                // with the current value of beta

    // Create a copy of Ystd to use for updating Y_l
    arma::mat Y_copy = Ystd;

    // Update each coefficient for each predictor
    // using cyclical coordinate descent
    for (int j = 0; j < q; j++) {
      arma::vec Xj = Xstd.col(j);
      for (int l = 0; l < k; l++) {
        arma::vec Y_l = Y_copy.col(l);
        double rho = dot(Xj, Y_l - Xstd * beta.col(l) + beta(j, l) * Xj);
        double z = dot(Xj, Xj);
        double c = 0;

        if (rho < -lambda / 2) {
          c = (rho + lambda / 2) / z;
        } else if (rho > lambda / 2) {
          c = (rho - lambda / 2) / z;
        } else {
          c = 0;
        }
        beta(j, l) = c;

        // Update Y_l for the next iteration
        Y_l = Y_l - (Xj * (beta(j, l) - beta_old(j, l)));
      }
    }

    // Check convergence
    if (iter > 0) {
      if (arma::all(arma::vectorise(arma::abs(beta - beta_old)) < tol)) {
        break;  // Converged, exit the loop
      }
    }

    // If the loop reaches the last iteration and has not broken
    // (not converged),
    // emit a warning
    if (iter == max_iter - 1) {
      Rcpp::warning(
          "The algorithm did not converge within the specified maximum number "
          "of iterations.");
    }
  }

  return beta.t();
}
