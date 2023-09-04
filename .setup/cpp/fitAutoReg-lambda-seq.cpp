// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-lambda-seq.cpp
// Ivan Jacob Agaloos Pesigan
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Function to generate the sequence of lambdas
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @inheritParams FitVARLasso
//' @param n_lambdas Integer.
//'   Number of lambdas to generate.
//'
//' @return Returns a vector of lambdas.
//'
//' @examples
//' YStd <- StdMat(dat_p2_yx$Y)
//' XStd <- StdMat(dat_p2_yx$X[, -1]) # remove the constant column
//' LambdaSeq(YStd = YStd, XStd = XStd, n_lambdas = 100)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::vec LambdaSeq(const arma::mat& YStd, const arma::mat& XStd,
                    int n_lambdas) {
  // Step 1: Determine the number of outcome variables
  int num_outcome_vars = YStd.n_cols;

  // Step 2: Compute the product of the transpose of XStd and XStd
  arma::mat XtX = trans(XStd) * XStd;

  // Step 3: Calculate the maximum lambda value for Lasso regularization
  double lambda_max = arma::max(diagvec(XtX)) / (num_outcome_vars * 2);

  // Step 4: Compute the logarithm of lambda_max
  double log_lambda_max = std::log10(lambda_max);

  // Step 5: Initialize a vector 'lambda_seq' to store the sequence of lambda
  // values
  arma::vec lambda_seq(n_lambdas);

  // Step 6: Calculate the step size for logarithmic lambda values
  double log_lambda_step =
      (std::log10(lambda_max / 1000) - log_lambda_max) / (n_lambdas - 1);

  // Step 7: Generate the sequence of lambda values using a logarithmic scale
  for (int i = 0; i < n_lambdas; ++i) {
    double log_lambda = log_lambda_max + i * log_lambda_step;
    lambda_seq(i) = std::pow(10, log_lambda);
  }

  // Step 8: Return the sequence of lambda values
  return lambda_seq;
}
