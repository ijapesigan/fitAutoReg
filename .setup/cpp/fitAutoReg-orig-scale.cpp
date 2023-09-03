// -----------------------------------------------------------------------------
// edit .setup/cpp/008-fitAutoReg-orig-scale.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Return Standardized Estimates to the Original Scale
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef_std Numeric matrix.
//'   Standardized estimates of the autoregression
//'   and cross regression coefficients.
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//'
//' @examples
//' Y <- dat_p2_yx$Y
//' X <- dat_p2_yx$X[, -1] # remove the constant column
//' YStd <- StdMat(Y)
//' XStd <- StdMat(X)
//' coef_std <- FitVAROLS(Y = YStd, X = XStd)
//' FitVAROLS(Y = Y, X = X)
//' OrigScale(coef_std = coef_std, Y = Y, X = X)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat OrigScale(const arma::mat& coef_std, const arma::mat& Y, const arma::mat& X) {
  // Step 1: Get the number of outcome variables and predictor variables
  int num_outcome_vars = coef_std.n_rows;
  int num_predictor_vars = coef_std.n_cols;

  // Step 2: Initialize vectors to store standard deviations of outcome (Y) and predictor (X) variables
  arma::vec sd_Y(num_outcome_vars);
  arma::vec sd_X(num_predictor_vars);

  // Step 3: Calculate the standard deviation for each outcome variable (Y)
  for (int l = 0; l < num_outcome_vars; l++) {
    sd_Y(l) = arma::as_scalar(arma::stddev(Y.col(l), 0, 0));
  }

  // Step 4: Calculate the standard deviation for each predictor variable (X)
  for (int j = 0; j < num_predictor_vars; j++) {
    sd_X(j) = arma::as_scalar(arma::stddev(X.col(j), 0, 0));
  }

  // Step 5: Initialize a matrix 'orig' to store coefficients in the original scale
  arma::mat orig(num_outcome_vars, num_predictor_vars);

  // Step 6: Compute original-scale coefficients by scaling back from standardized coefficients
  for (int l = 0; l < num_outcome_vars; l++) {
    for (int j = 0; j < num_predictor_vars; j++) {
      double orig_coef = coef_std(l, j) * sd_Y(l) / sd_X(j);
      orig(l, j) = orig_coef;
    }
  }

  // Step 7: Return the coefficients in the original scale
  return orig;
}
