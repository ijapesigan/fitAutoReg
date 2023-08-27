// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso-search.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters using Lasso Regularization
//' with Lambda Search
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Ystd Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param Xstd Numeric matrix.
//'   Matrix of standardized predictors (X).
//' @param lambdas Numeric vector.
//'   Vector of lambda hyperparameters for Lasso regularization.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm
//'   (e.g., `max_iter = 10000`).
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance
//'   (e.g., `tol = 1e-5`).
//' @param crit Character string.
//'   Information criteria to use.
//'   Valid values include `"aic"`, `"bic"`, and `"ebic"`.
//'
//' @return Matrix of estimated autoregressive
//' and cross-regression coefficients.
//'
//' @examples
//' Ystd <- StdMat(dat_p2_yx$Y)
//' Xstd <- StdMat(dat_p2_yx$X[, -1])
//' lambdas <- LambdaSeq(Y = Ystd, X = Xstd, n_lambdas = 100)
//' FitVARLassoSearch(Ystd = Ystd, Xstd = Xstd, lambdas = lambdas,
//'   crit = "ebic", max_iter = 1000, tol = 1e-5)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLassoSearch(const arma::mat& Ystd, const arma::mat& Xstd,
                            const arma::vec& lambdas, const std::string& crit,
                            int max_iter, double tol) {
  int n = Xstd.n_rows;  // Number of observations (rows in X)
  int q = Xstd.n_cols;  // Number of columns in X (predictors)

  // Variables to track the minimum criterion value
  double min_criterion = std::numeric_limits<double>::infinity();
  arma::mat beta_min_criterion;

  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Fit the VAR model using Lasso regularization
    arma::mat beta = FitVARLasso(Ystd, Xstd, lambda, max_iter, tol);

    // Calculate the residuals
    arma::mat residuals = Ystd - Xstd * beta.t();

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
      beta_min_criterion = beta;
    }
  }

  return beta_min_criterion;
}
