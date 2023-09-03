// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso-search.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters
//' using Lasso Regularization with Lambda Search
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param lambdas Numeric vector.
//'   Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param crit Character string.
//'   Information criteria to use.
//'   Valid values include `"aic"`, `"bic"`, and `"ebic"`.
//' @inheritParams FitVARLasso
//'
//' @return Matrix of estimated autoregressive
//'   and cross-regression coefficients.
//'
//' @examples
//' YStd <- StdMat(dat_p2_yx$Y)
//' XStd <- StdMat(dat_p2_yx$X[, -1]) # remove the constant column
//' lambdas <- LambdaSeq(
//'   YStd = YStd,
//'   XStd = XStd,
//'   n_lambdas = 100
//' )
//' FitVARLassoSearch(
//'   YStd = YStd,
//'   XStd = XStd,
//'   lambdas = lambdas,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLassoSearch(const arma::mat& YStd, const arma::mat& XStd,
                            const arma::vec& lambdas, const std::string& crit,
                            int max_iter, double tol) {
  // Step 1: Get the number of time points and predictor variables
  int time = XStd.n_rows;
  int num_predictor_vars = XStd.n_cols;

  // Step 2: Initialize variables to keep track of the best model based on the
  // selected criterion
  double min_criterion = std::numeric_limits<double>::infinity();
  arma::mat coef_min_crit;

  // Step 3: Loop over each lambda value in the 'lambdas' vector
  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Step 4: Fit a VAR model with Lasso regularization for the current lambda
    // value
    arma::mat coef = FitVARLasso(YStd, XStd, lambda, max_iter, tol);

    // Step 5: Calculate the residuals, RSS, and the number of nonzero
    // parameters
    arma::mat residuals = YStd - XStd * coef.t();
    double rss = arma::accu(residuals % residuals);
    int num_params = arma::sum(arma::vectorise(coef != 0));

    // Step 6: Calculate the chosen information criterion (AIC, BIC, or EBIC)
    double aic = time * std::log(rss / time) + 2.0 * num_params;
    double bic = time * std::log(rss / time) + num_params * std::log(time);
    double ebic =
        time * std::log(rss / time) +
        2.0 * num_params * std::log(time / double(num_predictor_vars));

    // Step 7: Determine the current criterion value based on user choice
    // ('aic', 'bic', or 'ebic')
    double current_criterion = 0.0;
    if (crit == "aic") {
      current_criterion = aic;
    } else if (crit == "bic") {
      current_criterion = bic;
    } else if (crit == "ebic") {
      current_criterion = ebic;
    }

    // Step 8: Update the best model if the current criterion is smaller than
    // the minimum
    if (current_criterion < min_criterion) {
      min_criterion = current_criterion;
      coef_min_crit = coef;
    }
  }

  // Step 9: Return the coefficients of the best model based on the selected
  // criterion
  return coef_min_crit;
}
