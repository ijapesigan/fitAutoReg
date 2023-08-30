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
  // Order of the VAR model (number of lags)
  int p = coef.n_cols / constant.n_elem;

  // Simulate data
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_removed = X.cols(1, X.n_cols - 1);

  // OLS
  arma::mat ols = FitVAROLS(Y, X);
  arma::vec const_b = ols.col(0);  // OLS constant vector

  // Standardize
  arma::mat XStd = StdMat(X_removed);
  arma::mat YStd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Lasso
  arma::mat coef_std_b =
      FitVARLassoSearch(YStd, XStd, lambdas, crit, max_iter, tol);

  // Original scale
  arma::mat coef_orig = OrigScale(coef_std_b, Y, X_removed);

  // OLS constant and Lasso coefficient matrix
  arma::mat coef_b = arma::join_horiz(const_b, coef_orig);

  return arma::vectorise(coef_b);
}
