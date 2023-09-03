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
//' @inheritParams FitVARLassoSearch
//'
//' @return List with the following elements:
//'   - **criteria**: Matrix with columns for
//'     lambda, AIC, BIC, and EBIC values.
//'   - **fit**: List of matrices containing
//'     the estimated autoregressive
//'     and cross-regression coefficients for each lambda.
//'
//' @examples
//' YStd <- StdMat(dat_p2_yx$Y)
//' XStd <- StdMat(dat_p2_yx$X[, -1])
//' lambdas <- 10^seq(-5, 5, length.out = 100)
//' search <- SearchVARLasso(YStd = YStd, XStd = XStd, lambdas = lambdas,
//'   max_iter = 10000, tol = 1e-5)
//' plot(x = 1:nrow(search$criteria), y = search$criteria[, 4],
//'   type = "b", xlab = "lambda", ylab = "EBIC")
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
Rcpp::List SearchVARLasso(const arma::mat& YStd, const arma::mat& XStd,
                          const arma::vec& lambdas, int max_iter, double tol) {
  // Step 1: Get the number of time points and predictor variables
  int time = XStd.n_rows;
  int num_predictor_vars = XStd.n_cols;

  // Step 2: Initialize matrices to store results and a list to store fitted
  // models
  arma::mat results(lambdas.n_elem, 4, arma::fill::zeros);
  Rcpp::List fit_list(lambdas.n_elem);

  // Step 3: Loop over each lambda value in the 'lambdas' vector
  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Step 4: Fit a VAR model with Lasso regularization for the current lambda
    // value
    arma::mat beta = FitVARLasso(YStd, XStd, lambda, max_iter, tol);

    // Step 5: Calculate the residuals, RSS, and the number of nonzero
    // parameters
    arma::mat residuals = YStd - XStd * beta.t();
    double rss = arma::accu(residuals % residuals);
    int num_params = arma::sum(arma::vectorise(beta != 0));

    // Step 6: Calculate the information criteria (AIC, BIC, and EBIC) for the
    // fitted model
    double aic = time * std::log(rss / time) + 2.0 * num_params;
    double bic = time * std::log(rss / time) + num_params * std::log(time);
    double ebic =
        time * std::log(rss / time) +
        2.0 * num_params * std::log(time / double(num_predictor_vars));

    // Step 7: Store the lambda value, AIC, BIC, and EBIC in the 'results'
    // matrix
    results(i, 0) = lambda;
    results(i, 1) = aic;
    results(i, 2) = bic;
    results(i, 3) = ebic;

    // Step 8: Store the fitted model (beta) in the 'fit_list'
    fit_list[i] = beta;
  }

  // Step 9: Return a list containing the criteria results and the list of
  // fitted models
  return Rcpp::List::create(Rcpp::Named("criteria") = results,
                            Rcpp::Named("fit") = fit_list);
}
