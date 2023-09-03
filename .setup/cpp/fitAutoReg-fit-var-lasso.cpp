// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters
//' using Lasso Regularization
//'
//' This function estimates the parameters of a VAR model
//' using the Lasso regularization method
//' with cyclical coordinate descent.
//' The Lasso method is used to estimate the autoregressive
//' and cross-regression coefficients with sparsity.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param YStd Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param XStd Numeric matrix.
//'   Matrix of standardized predictors (X).
//'   `XStd` should not include a vector of ones in column one.
//' @param lambda Numeric.
//'   Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param max_iter Integer.
//'   The maximum number of iterations
//'   for the coordinate descent algorithm
//'   (e.g., `max_iter = 10000`).
//' @param tol Numeric.
//'   Convergence tolerance.
//'   The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance
//'   (e.g., `tol = 1e-5`).
//'
//' @return Matrix of estimated autoregressive
//'   and cross-regression coefficients.
//'
//' @examples
//' YStd <- StdMat(dat_p2_yx$Y)
//' XStd <- StdMat(dat_p2_yx$X[, -1]) # remove the constant column
//' lambda <- 73.90722
//' FitVARLasso(
//'   YStd = YStd,
//'   XStd = XStd,
//'   lambda = lambda,
//'   max_iter = 10000,
//'   tol = 1e-5
//' )
//'
//' @details
//' The [fitAutoReg::FitVARLasso()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Lasso regularization method.
//' Given the input matrices `YStd` and `XStd`,
//' where `YStd` is the matrix of standardized dependent variables,
//' and `XStd` is the matrix of standardized predictors,
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
//'   (columns of `XStd`),
//'   while the inner loop iterates over the outcome variables
//'   (columns of `YStd`).
//' - **Coefficient Update**: For each predictor variable (column of `XStd`),
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
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLasso(const arma::mat& YStd, const arma::mat& XStd, const double& lambda, int max_iter, double tol) {
  // Step 1: Determine the number of predictor variables and outcome variables
  int num_predictor_vars = XStd.n_cols;
  int num_outcome_vars = YStd.n_cols;

  // Step 2: Initialize matrices to store QR decomposition results
  arma::mat Q, R;

  // Step 3: Perform QR decomposition of the standardized predictor matrix XStd
  arma::qr_econ(Q, R, XStd);

  // Step 4: Initialize a matrix 'coef' to store the estimated coefficients
  arma::mat coef = arma::solve(R, Q.t() * YStd);

  // Step 5: Iterate for a maximum of 'max_iter' times
  for (int iter = 0; iter < max_iter; iter++) {
    // Step 5.1: Create a copy of the current 'coef' matrix for comparison
    arma::mat coef_old = coef;

    // Step 5.2: Create a copy of the outcome matrix 'YStd' for updates
    arma::mat Y_copy = YStd;

    // Step 5.3: Loop over predictor variables
    for (int j = 0; j < num_predictor_vars; j++) {
      // Step 5.3.1: Extract the j-th column of the standardized predictor matrix XStd
      arma::vec Xj = XStd.col(j);

      // Step 5.3.2: Loop over outcome variables
      for (int l = 0; l < num_outcome_vars; l++) {
        // Step 5.3.2.1: Extract the l-th column of the copy of the outcome matrix Y_copy
        arma::vec Y_l = Y_copy.col(l);

        // Step 5.3.2.2: Compute 'rho' and 'z' for Lasso regularization
        double rho = dot(Xj, Y_l - XStd * coef.col(l) + coef(j, l) * Xj);
        double z = dot(Xj, Xj);

        // Step 5.3.2.3: Apply Lasso regularization and update the 'coef' matrix
        double c = 0;
        if (rho < -lambda / 2) {
          c = (rho + lambda / 2) / z;
        } else if (rho > lambda / 2) {
          c = (rho - lambda / 2) / z;
        } else {
          c = 0;
        }
        coef(j, l) = c;

        // Step 5.3.2.4: Update the l-th column of the copy of the outcome matrix Y_copy
        Y_l = Y_l - (Xj * (coef(j, l) - coef_old(j, l)));
      }
    }

    // Step 5.4: Check for convergence based on the change in 'coef'
    if (iter > 0) {
      if (arma::all(arma::vectorise(arma::abs(coef - coef_old)) < tol)) {
        break;
      }
    }

    // Step 5.5: If maximum iterations are reached without convergence, issue a warning
    if (iter == max_iter - 1) {
      Rcpp::warning(
          "The algorithm did not converge within the specified maximum number "
          "of iterations.");
    }
  }

  // Step 6: Return the estimated coefficients (transposed for the desired format)
  return coef.t();
}
