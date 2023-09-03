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
  // Step 1: Get the number of rows (n) and columns (num_vars) in the input matrix X
  int n = X.n_rows;
  int num_vars = X.n_cols;

  // Step 2: Initialize the standardized matrix XStd with zeros
  arma::mat XStd(n, num_vars, arma::fill::zeros);

  // Step 3: Calculate column means and store them in the col_means vector
  arma::vec col_means(num_vars, arma::fill::zeros);
  for (int j = 0; j < num_vars; j++) {
    for (int i = 0; i < n; i++) {
      col_means(j) += X(i, j);
    }
    col_means(j) /= n;  // Calculate the mean for column j
  }

  // Step 4: Calculate column standard deviations and store them in the col_sds vector
  arma::vec col_sds(num_vars, arma::fill::zeros);
  for (int j = 0; j < num_vars; j++) {
    for (int i = 0; i < n; i++) {
      col_sds(j) += std::pow(X(i, j) - col_means(j), 2);
    }
    col_sds(j) = std::sqrt(col_sds(j) / (n - 1));  // Calculate the standard deviation for column j
  }

  // Step 5: Standardize the matrix X by subtracting column means and dividing by column standard deviations
  for (int j = 0; j < num_vars; j++) {
    for (int i = 0; i < n; i++) {
      XStd(i, j) = (X(i, j) - col_means(j)) / col_sds(j);  // Standardize each element of X
    }
  }

  // Step 6: Return the standardized matrix XStd
  return XStd;
}
