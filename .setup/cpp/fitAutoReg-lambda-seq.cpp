// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-lambda-seq.cpp
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
  int k = YStd.n_cols;  // Number of variables

  arma::mat XtX = trans(XStd) * XStd;
  double lambda_max = arma::max(diagvec(XtX)) / (k * 2);

  // Generate the sequence of lambdas
  double log_lambda_max = std::log10(lambda_max);
  arma::vec lambda_seq(n_lambdas);
  double log_lambda_step =
      (std::log10(lambda_max / 1000) - log_lambda_max) / (n_lambdas - 1);

  for (int i = 0; i < n_lambdas; ++i) {
    double log_lambda = log_lambda_max + i * log_lambda_step;
    lambda_seq(i) = std::pow(10, log_lambda);
  }

  return lambda_seq;
}
