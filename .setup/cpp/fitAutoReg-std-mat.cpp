// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-std-mat.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Standardize Matrix
//'
//' This function standardizes the given matrix by centering the columns
//' and scaling them to have unit variance.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param X Numeric matrix.
//'   The matrix to be standardized.
//'
//' @return Numeric matrix with standardized values.
//'
//' @examples
//' std <- StdMat(dat_p2)
//' colMeans(std)
//' var(std)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat StdMat(const arma::mat& X) {
  int q = X.n_cols;  // Number of predictors
  int n = X.n_rows;  // Number of observations

  // Initialize the standardized matrix
  arma::mat XStd(n, q, arma::fill::zeros);

  // Calculate column means
  arma::vec col_means(q, arma::fill::zeros);
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      col_means(j) += X(i, j);
    }
    col_means(j) /= n;
  }

  // Calculate column standard deviations
  arma::vec col_sds(q, arma::fill::zeros);
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      col_sds(j) += std::pow(X(i, j) - col_means(j), 2);
    }
    col_sds(j) = std::sqrt(col_sds(j) / (n - 1));
  }

  // Standardize the matrix
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      XStd(i, j) = (X(i, j) - col_means(j)) / col_sds(j);
    }
  }

  return XStd;
}
