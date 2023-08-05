// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-search-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Compute AIC, BIC, and EBIC for Lasso Regularization
//'
//' This function computes the Akaike Information Criterion (AIC),
//' Bayesian Information Criterion (BIC),
//' and Extended Bayesian Information Criterion (EBIC)
//' for a given matrix of predictors `X`, a matrix of outcomes `Y`,
//' and a vector of lambda hyperparameters for Lasso regularization.
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
//'
//' @return List containing two elements:
//'   - Element 1: Matrix with columns for
//'     lambda, AIC, BIC, and EBIC values.
//'   - Element 2: List of matrices containing
//'     the estimated autoregressive
//'     and cross-regression coefficients for each lambda.
//'
//' @examples
//' Ystd <- StdMat(dat_demo_yx$Y)
//' Xstd <- StdMat(dat_demo_yx$X[, -1])
//' lambdas <- 10^seq(-5, 5, length.out = 100)
//' search <- SearchVARLasso(Ystd = Ystd, Xstd = Xstd, lambdas = lambdas,
//'   max_iter = 10000, tol = 1e-5)
//' plot(x = 1:nrow(search$criteria), y = search$criteria[, 4],
//'   type = "b", xlab = "lambda", ylab = "EBIC")
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
Rcpp::List SearchVARLasso(const arma::mat& Ystd, const arma::mat& Xstd,
                          const arma::vec& lambdas, int max_iter, double tol) {
  int n = Xstd.n_rows;  // Number of observations (rows in X)
  int q = Xstd.n_cols;  // Number of columns in X (predictors)

  // Armadillo matrix to store the lambda, AIC, BIC, and EBIC values
  arma::mat results(lambdas.n_elem, 4, arma::fill::zeros);

  // List to store the output of FitVARLasso for each lambda
  Rcpp::List fit_list(lambdas.n_elem);

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

    // Store the lambda, AIC, BIC, and EBIC values in the results matrix
    results(i, 0) = lambda;
    results(i, 1) = aic;
    results(i, 2) = bic;
    results(i, 3) = ebic;

    // Store the output of FitVARLasso for this lambda in the fit_list
    fit_list[i] = beta;
  }

  return Rcpp::List::create(Rcpp::Named("criteria") = results,
                            Rcpp::Named("fit") = fit_list);
}
