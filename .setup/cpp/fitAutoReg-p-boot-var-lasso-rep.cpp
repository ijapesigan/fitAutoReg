// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-lasso-rep.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVARLassoRep(int time, int burn_in, const arma::vec& constant,
                           const arma::mat& coef, const arma::mat& chol_cov,
                           int n_lambdas, const std::string& crit, int max_iter,
                           double tol) {
  // Indices
  int k = constant.n_elem;  // Number of variables
  int q = coef.n_cols;      // Dimension of the coefficient matrix
  int p = q / k;            // Order of the VAR model (number of lags)

  // Simulate data
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_removed = X.cols(1, X.n_cols - 1);

  // OLS
  arma::mat ols = FitVAROLS(Y, X);
  arma::vec pb_const = ols.col(0);  // OLS constant vector

  // Standardize
  arma::mat Xstd = StdMat(X_removed);
  arma::mat Ystd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(Ystd, Xstd, n_lambdas);

  // Lasso
  arma::mat pb_std =
      FitVARLassoSearch(Ystd, Xstd, lambdas, crit, max_iter, tol);

  // Original scale
  arma::mat pb_orig = OrigScale(pb_std, Y, X_removed);

  // OLS constant and Lasso coefficient matrix
  arma::mat pb_coef = arma::join_horiz(pb_const, pb_orig);

  return arma::vectorise(pb_coef);
}
