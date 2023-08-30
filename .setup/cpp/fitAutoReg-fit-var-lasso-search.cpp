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
  int n = XStd.n_rows;  // Number of observations (rows in X)
  int q = XStd.n_cols;  // Number of columns in X (predictors)

  // Variables to track the minimum criterion value
  double min_criterion = std::numeric_limits<double>::infinity();
  arma::mat beta_min_crit;

  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Fit the VAR model using Lasso regularization
    arma::mat beta = FitVARLasso(YStd, XStd, lambda, max_iter, tol);

    // Calculate the residuals
    arma::mat residuals = YStd - XStd * beta.t();

    // Compute the residual sum of squares (RSS)
    double rss = arma::accu(residuals % residuals);

    // Compute the degrees of freedom for each parameter
    int num_params = arma::sum(arma::vectorise(beta != 0));

    // Compute the AIC, BIC, and EBIC criteria
    double aic = n * std::log(rss / n) + 2.0 * num_params;
    double bic = n * std::log(rss / n) + num_params * std::log(n);
    double ebic =
        n * std::log(rss / n) + 2.0 * num_params * std::log(n / double(q));

    // Update the minimum criterion and its index if necessary
    double current_criterion = 0.0;
    if (crit == "aic") {
      current_criterion = aic;
    } else if (crit == "bic") {
      current_criterion = bic;
    } else if (crit == "ebic") {
      current_criterion = ebic;
    }

    if (current_criterion < min_criterion) {
      min_criterion = current_criterion;
      beta_min_crit = beta;
    }
  }

  return beta_min_crit;
}
