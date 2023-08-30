#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov);

Rcpp::List YX(const arma::mat& data, int p);

arma::mat OrigScale(const arma::mat& coef_std, const arma::mat& Y,
                    const arma::mat& X);

arma::mat StdMat(const arma::mat& X);

arma::mat FitVAROLS(const arma::mat& Y, const arma::mat& X);

arma::vec LambdaSeq(const arma::mat& Y, const arma::mat& X, int n_lambdas);

arma::mat FitVARLasso(const arma::mat& Ystd, const arma::mat& Xstd,
                      const double& lambda, int max_iter, double tol);

arma::mat FitVARLassoSearch(const arma::mat& Ystd, const arma::mat& Xstd,
                            const arma::vec& lambdas, const std::string& crit,
                            int max_iter, double tol);

Rcpp::List SearchVARLasso(const arma::mat& Ystd, const arma::mat& Xstd,
                          const arma::vec& lambdas, int max_iter, double tol);

arma::vec PBootVARLassoRep(int time, int burn_in, const arma::vec& constant,
                           const arma::mat& coef, const arma::mat& chol_cov,
                           int n_lambdas, const std::string& crit, int max_iter,
                           double tol);

arma::mat PBootVARLassoSim(int B, int time, int burn_in,
                           const arma::vec& constant, const arma::mat& coef,
                           const arma::mat& chol_cov, int n_lambdas,
                           const std::string& crit, int max_iter, double tol);

Rcpp::List PBootVARLasso(const arma::mat& data, int p, int B, int burn_in,
                         int n_lambdas, const std::string& crit, int max_iter,
                         double tol);

arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant,
                         const arma::mat& coef, const arma::mat& chol_cov);

arma::mat PBootVAROLSSim(int B, int time, int burn_in,
                         const arma::vec& constant, const arma::mat& coef,
                         const arma::mat& chol_cov);

Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B, int burn_in);

Rcpp::List RBootVAROLS(const arma::mat& data, int p, int B);

Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B, int n_lambdas,
                         const std::string& crit, int max_iter, double tol);
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
//'   `Xstd` should not include a vector of ones in column one.
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
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters using Lasso Regularization
//'
//' This function estimates the parameters of a VAR model
//' using the Lasso regularization method with cyclical coordinate descent.
//' The Lasso method is used to estimate the autoregressive
//' and cross-regression coefficients with sparsity.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Ystd Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param Xstd Numeric matrix.
//'   Matrix of standardized predictors (X).
//'   `Xstd` should not include a vector of ones in column one.
//' @param lambda Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param max_iter Integer.
//'   The maximum number of iterations for the coordinate descent algorithm
//'   (e.g., `max_iter = 10000`).
//' @param tol Numeric.
//'   Convergence tolerance. The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance
//'   (e.g., `tol = 1e-5`).
//'
//' @return Matrix of estimated autoregressive and
//' cross-regression coefficients.
//'
//' @examples
//' Ystd <- StdMat(dat_p2_yx$Y)
//' Xstd <- StdMat(dat_p2_yx$X[, -1])
//' lambda <- 73.90722
//' FitVARLasso(Ystd = Ystd, Xstd = Xstd, lambda = lambda,
//'   max_iter = 10000, tol = 1e-5)
//'
//' @details
//' The [fitAutoReg::FitVARLasso()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Lasso regularization method.
//' Given the input matrices `Ystd` and `Xstd`,
//' where `Ystd` is the matrix of standardized dependent variables,
//' and `Xstd` is the matrix of standardized predictors,
//' the function computes the autoregressive and cross-regression coefficients
//' of the VAR model with sparsity induced by the Lasso regularization.
//'
//' The steps involved in estimating the VAR model parameters
//' using Lasso are as follows:
//'
//' - **Initialization**: The function initializes the coefficient matrix
//'   `beta` with OLS estimates.
//'   The `beta` matrix will store the estimated autoregressive and
//'   cross-regression coefficients.
//' - **Coordinate Descent Loop**: The function performs
//'   the cyclical coordinate descent algorithm
//'   to estimate the coefficients iteratively.
//'   The loop iterates `max_iter` times,
//'   or until convergence is achieved.
//'   The outer loop iterates over the predictor variables
//'   (columns of `Xstd`),
//'   while the inner loop iterates over the outcome variables
//'   (columns of `Ystd`).
//' - **Coefficient Update**: For each predictor variable (column of `Xstd`),
//'   the function iteratively updates the corresponding column of `beta`
//'   using the coordinate descent algorithm with L1 norm regularization
//'   (Lasso).
//'   The update involves calculating the soft-thresholded value `c`,
//'   which encourages sparsity in the coefficients.
//'   The algorithm continues until the change in coefficients
//'   between iterations is below the specified tolerance `tol`
//'   or when the maximum number of iterations is reached.
//' - **Convergence Check**: The function checks for convergence
//'   by comparing the current `beta`
//'   matrix with the previous iteration's `beta_old`.
//'   If the maximum absolute difference between `beta` and `beta_old`
//'   is below the tolerance `tol`,
//'   the algorithm is considered converged, and the loop exits.
//'
//' @seealso
//' The [fitAutoReg::FitVAROLS()] function for estimating VAR model parameters
//' using OLS.
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLasso(const arma::mat& Ystd, const arma::mat& Xstd,
                      const double& lambda, int max_iter, double tol) {
  int q = Xstd.n_cols;  // Number of predictors (excluding the intercept column)
  int k = Ystd.n_cols;  // Number of outcomes

  // OLS starting values
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr(Q, R, Xstd);
  // Solve the linear system R * beta = Q.t() * Ystd
  arma::mat beta = arma::solve(R, Q.t() * Ystd);

  // Coordinate Descent Loop
  for (int iter = 0; iter < max_iter; iter++) {
    arma::mat beta_old = beta;  // Initialize beta_old
                                // with the current value of beta

    // Create a copy of Ystd to use for updating Y_l
    arma::mat Y_copy = Ystd;

    // Update each coefficient for each predictor
    // using cyclical coordinate descent
    for (int j = 0; j < q; j++) {
      arma::vec Xj = Xstd.col(j);
      for (int l = 0; l < k; l++) {
        arma::vec Y_l = Y_copy.col(l);
        double rho = dot(Xj, Y_l - Xstd * beta.col(l) + beta(j, l) * Xj);
        double z = dot(Xj, Xj);
        double c = 0;

        if (rho < -lambda / 2) {
          c = (rho + lambda / 2) / z;
        } else if (rho > lambda / 2) {
          c = (rho - lambda / 2) / z;
        } else {
          c = 0;
        }
        beta(j, l) = c;

        // Update Y_l for the next iteration
        Y_l = Y_l - (Xj * (beta(j, l) - beta_old(j, l)));
      }
    }

    // Check convergence
    if (iter > 0) {
      if (arma::all(arma::vectorise(arma::abs(beta - beta_old)) < tol)) {
        break;  // Converged, exit the loop
      }
    }

    // If the loop reaches the last iteration and has not broken
    // (not converged),
    // emit a warning
    if (iter == max_iter - 1) {
      Rcpp::warning(
          "The algorithm did not converge within the specified maximum number "
          "of iterations.");
    }
  }

  return beta.t();
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters using OLS
//'
//' This function estimates the parameters of a VAR model
//' using the Ordinary Least Squares (OLS) method.
//' The OLS method is used to estimate the autoregressive
//' and cross-regression coefficients.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//'
//' @return Matrix of estimated autoregressive
//' and cross-regression coefficients.
//'
//' @examples
//' Y <- dat_p2_yx$Y
//' X <- dat_p2_yx$X
//' FitVAROLS(Y = Y, X = X)
//'
//' @details
//' The [fitAutoReg::FitVAROLS()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Ordinary Least Squares (OLS) method.
//' Given the input matrices `Y` and `X`,
//' where `Y` is the matrix of dependent variables,
//' and `X` is the matrix of predictors,
//' the function computes the autoregressive and cross-regression coefficients
//' of the VAR model.
//' Note that if the first column of `X` is a vector of ones,
//' the constant vector is also estimated.
//'
//' The steps involved in estimating the VAR model parameters
//' using OLS are as follows:
//'
//' - Compute the QR decomposition of the lagged predictor matrix `X`
//'   using the `qr` function from the Armadillo library.
//' - Extract the `Q` and `R` matrices from the QR decomposition.
//' - Solve the linear system `R * coef = Q.t() * Y`
//'   to estimate the VAR model coefficients `coef`.
//' - The function returns a matrix containing the estimated
//'   autoregressive and cross-regression coefficients of the VAR model.
//'
//' @seealso
//' The `qr_econ` function from the Armadillo library for QR decomposition.
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVAROLS(const arma::mat& Y, const arma::mat& X) {
  // Estimate VAR model parameters using QR decomposition
  arma::mat Q, R;
  arma::qr_econ(Q, R, X);

  // Solve the linear system R * coef = Q.t() * Y
  arma::mat coef = arma::solve(R, Q.t() * Y);

  return coef.t();
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-lambda-seq.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Function to generate the sequence of lambdas
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//' @param n_lambdas Integer.
//'   Number of lambdas to generate.
//'
//' @return Returns a vector of lambdas.
//'
//' @examples
//' Ystd <- StdMat(dat_p2_yx$Y)
//' Xstd <- StdMat(dat_p2_yx$X[, -1])
//' LambdaSeq(Y = Ystd, X = Xstd, n_lambdas = 100)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::vec LambdaSeq(const arma::mat& Y, const arma::mat& X, int n_lambdas) {
  int k = Y.n_cols;  // Number of variables

  arma::mat XtX = trans(X) * X;
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
// -----------------------------------------------------------------------------
// edit .setup/cpp/008-fitAutoReg-orig-scale.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Return Standardized Estimates to the Original Scale
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param coef_std Numeric matrix.
//'   Standardized estimates of the autoregression
//'   and cross regression coefficients.
//' @param Y Numeric matrix.
//'   Matrix of dependent variables (Y).
//' @param X Numeric matrix.
//'   Matrix of predictors (X).
//'
//' @examples
//' Y <- dat_p2_yx$Y
//' X <- dat_p2_yx$X[, -1]
//' Ystd <- StdMat(Y)
//' Xstd <- StdMat(X)
//' coef_std <- FitVAROLS(Y = Ystd, X = Xstd)
//' FitVAROLS(Y = Y, X = X)
//' OrigScale(coef_std = coef_std, Y = Y, X = X)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat OrigScale(const arma::mat& coef_std, const arma::mat& Y,
                    const arma::mat& X) {
  int k = coef_std.n_rows;  // Number of outcomes
  int q = coef_std.n_cols;  // Number of predictors

  arma::vec sd_Y(k);
  arma::vec sd_X(q);

  // Calculate standard deviations of Y and X columns
  for (int l = 0; l < k; l++) {
    sd_Y(l) = arma::as_scalar(arma::stddev(Y.col(l), 0, 0));
  }
  for (int j = 0; j < q; j++) {
    sd_X(j) = arma::as_scalar(arma::stddev(X.col(j), 0, 0));
  }

  arma::mat coef_orig(k, q);
  for (int l = 0; l < k; l++) {
    for (int j = 0; j < q; j++) {
      double orig_coeff = coef_std(l, j) * sd_Y(l) / sd_X(j);
      coef_orig(l, j) = orig_coeff;
    }
  }

  return coef_orig;
}
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
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-lasso-sim.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Function to generate VAR time series data and fit VAR model B times
arma::mat PBootVARLassoSim(int B, int time, int burn_in,
                           const arma::vec& constant, const arma::mat& coef,
                           const arma::mat& chol_cov, int n_lambdas,
                           const std::string& crit, int max_iter, double tol) {
  int num_coef = constant.n_elem + coef.n_elem;
  arma::mat result(B, num_coef, arma::fill::zeros);

  for (int i = 0; i < B; i++) {
    arma::vec coef_est =
        PBootVARLassoRep(time, burn_in, constant, coef, chol_cov, n_lambdas,
                         crit, max_iter, tol);
    result.row(i) = arma::trans(coef_est);
  }

  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Parametric Bootstrap for the Vector Autoregressive Model
//' Using Lasso Regularization
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results
//'   in the simulation step.
//' @param n_lambdas Integer.
//'   Number of lambdas to generate.
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
//' @return List containing the estimates (`est`)
//' and bootstrap estimates (`boot`).
//'
//' @examples
//' pb <- PBootVARLasso(data = dat_p2, p = 2, B = 10, burn_in = 20,
//'   n_lambdas = 100, crit = "ebic", max_iter = 1000, tol = 1e-5)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVARLasso(const arma::mat& data, int p, int B, int burn_in,
                         int n_lambdas, const std::string& crit, int max_iter,
                         double tol) {
  // Indices
  int t = data.n_rows;  // Number of observations

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_removed = X.cols(1, X.n_cols - 1);

  // OLS
  arma::mat ols = FitVAROLS(Y, X);

  // Standardize
  arma::mat Xstd = StdMat(X_removed);
  arma::mat Ystd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(Ystd, Xstd, n_lambdas);

  // Lasso
  arma::mat pb_std = FitVARLassoSearch(Ystd, Xstd, lambdas, "ebic", 1000, 1e-5);

  // Set parameters
  arma::vec const_vec = ols.col(0);                      // OLS constant vector
  arma::mat coef_mat = OrigScale(pb_std, Y, X_removed);  // Lasso coefficients
  arma::mat coef =
      arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();
  // arma::mat residuals_tmp = Y.each_row() - const_vec.t();
  // arma::mat residuals = residuals_tmp - X_removed * coef_mat.t();

  // Calculate the covariance of residuals
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Result matrix
  arma::mat sim = PBootVARLassoSim(B, t, burn_in, const_vec, coef_mat, chol_cov,
                                   n_lambdas, crit, max_iter, tol);

  // Create a list to store the results
  Rcpp::List result;

  // Add coef as the first element
  result["est"] = coef;

  // Add sim as the second element
  result["boot"] = sim;

  // Return the list
  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols-rep.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Generate Data and Fit Model
arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant,
                         const arma::mat& coef, const arma::mat& chol_cov) {
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

  // OLS
  arma::mat pb_coef = FitVAROLS(Y, X);

  return arma::vectorise(pb_coef);
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols-sim.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Function to generate VAR time series data and fit VAR model B times
arma::mat PBootVAROLSSim(int B, int time, int burn_in,
                         const arma::vec& constant, const arma::mat& coef,
                         const arma::mat& chol_cov) {
  int num_coef = constant.n_elem + coef.n_elem;
  arma::mat result(B, num_coef, arma::fill::zeros);

  for (int i = 0; i < B; i++) {
    arma::vec coef_est =
        PBootVAROLSRep(time, burn_in, constant, coef, chol_cov);
    result.row(i) = arma::trans(coef_est);
  }

  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-p-boot-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Parametric Bootstrap for the Vector Autoregressive Model
//' Using Ordinary Least Squares
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results
//'   in the simulation step.
//'
//' @return List containing the estimates (`est`)
//' and bootstrap estimates (`boot`).
//'
//' @examples
//' pb <- PBootVAROLS(data = dat_p2, p = 2, B = 10, burn_in = 20)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B, int burn_in) {
  // Indices
  int t = data.n_rows;  // Number of observations

  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Set parameters
  arma::vec const_vec = coef.col(0);
  arma::mat coef_mat = coef.cols(1, coef.n_cols - 1);

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();

  // Calculate the covariance of residuals
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Result matrix
  arma::mat sim = PBootVAROLSSim(B, t, burn_in, const_vec, coef_mat, chol_cov);

  // Create a list to store the results
  Rcpp::List result;

  // Add coef as the first element
  result["est"] = coef;

  // Add sim as the second element
  result["boot"] = sim;

  // Return the list
  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-r-boot-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Residual Bootstrap for the Vector Autoregressive Model
//' Using Lasso Regularization
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//' @param n_lambdas Integer.
//'   Number of lambdas to generate.
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
//' @return List with the following elements:
//'   - List of bootstrap estimates
//'   - original `X`
//'   - List of bootstrapped `Y`
//'
//' @examples
//' pb <- RBootVARLasso(data = dat_p2, p = 2, B = 10,
//'   n_lambdas = 100, crit = "ebic", max_iter = 1000, tol = 1e-5)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B, int n_lambdas,
                         const std::string& crit, int max_iter, double tol) {
  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_removed = X.cols(1, X.n_cols - 1);

  // Indices
  int time = Y.n_rows;  // Number of observations

  // OLS
  arma::mat ols = FitVAROLS(Y, X);

  // Standardize
  arma::mat Xstd = StdMat(X_removed);
  arma::mat Ystd = StdMat(Y);

  // lambdas
  arma::vec lambdas = LambdaSeq(Ystd, Xstd, n_lambdas);

  // Lasso
  arma::mat coef_std =
      FitVARLassoSearch(Ystd, Xstd, lambdas, "ebic", 1000, 1e-5);
  arma::vec const_vec = ols.col(0);  // OLS constant vector
  arma::mat coef_mat = OrigScale(coef_std, Y, X_removed);  // Lasso coefficients
  arma::mat coef =
      arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Calculate the residuals
  arma::mat residuals = Y - X * coef.t();

  // Create a list to store bootstrap parameter estimates
  Rcpp::List coef_list(B);

  // Create a list of bootstrap Y
  Rcpp::List Y_list(B);

  for (int b = 0; b < B; ++b) {
    // Residual resampling
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // Simulate new data using bootstrapped residuals
    // and original parameter estimates
    arma::mat Y_b = X * coef.t() + residuals_b;

    // Fit VAR model using bootstrapped data
    arma::mat ols_b = FitVAROLS(Y_b, X);
    arma::mat Ystd_b = StdMat(Y);
    arma::mat coef_std_b =
        FitVARLassoSearch(Ystd_b, Xstd, lambdas, "ebic", 1000, 1e-5);

    // Original scale
    arma::vec const_vec_b = ols_b.col(0);
    arma::mat coef_mat_b = OrigScale(coef_std_b, Y_b, X_removed);
    arma::mat coef_b = arma::join_horiz(const_vec_b, coef_mat_b);

    // Store the bootstrapped parameter estimates in the list
    coef_list[b] = Rcpp::wrap(coef_b);

    // Store the bootstrapped Y in the list
    Y_list[b] = Rcpp::wrap(Y_b);
  }

  // Create a list to store the results
  Rcpp::List result;

  // Store bootstrap coefficients
  result["coef"] = coef_list;

  // Store regressors
  result["X"] = X;

  // Store bootstrap Y
  result["Y"] = Y_list;

  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-r-boot-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Residual Bootstrap for the Vector Autoregressive Model
//' Using Ordinary Least Squares
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//'
//' @return List with the following elements:
//'   - List of bootstrap estimates
//'   - original `X`
//'   - List of bootstrapped `Y`
//'
//' @examples
//' rb <- RBootVAROLS(data = dat_p2, p = 2, B = 10)
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVAROLS(const arma::mat& data, int p, int B) {
  // YX
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Indices
  int time = Y.n_rows;  // Number of observations

  // OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Residuals
  arma::mat residuals = Y - X * coef.t();

  // Create a list to store bootstrap parameter estimates
  Rcpp::List coef_list(B);

  // Create a list of bootstrap Y
  Rcpp::List Y_list(B);

  for (int b = 0; b < B; ++b) {
    // Residual resampling
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // Simulate new data using bootstrapped residuals
    // and original parameter estimates
    arma::mat Y_b = X * coef.t() + residuals_b;

    // Fit VAR model using bootstrapped data
    arma::mat coef_b = FitVAROLS(Y_b, X);

    // Store the bootstrapped parameter estimates in the list
    coef_list[b] = Rcpp::wrap(coef_b);

    // Store the bootstrapped Y in the list
    Y_list[b] = Rcpp::wrap(Y_b);
  }

  // Create a list to store the results
  Rcpp::List result;

  // Store bootstrap coefficients
  result["coef"] = coef_list;

  // Store regressors
  result["X"] = X;

  // Store bootstrap Y
  result["Y"] = Y_list;

  return result;
}
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
//' Ystd <- StdMat(dat_p2_yx$Y)
//' Xstd <- StdMat(dat_p2_yx$X[, -1])
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

  arma::mat X_std(n, q, arma::fill::zeros);  // Initialize the standardized
                                             // matrix

  // Calculate column means
  arma::vec col_means(q, arma::fill::zeros);
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      col_means(j) += X(i, j);
    }
    col_means(j) /= n;
  }

  // Calculate column standard deviations
  arma::vec col_stddevs(q, arma::fill::zeros);
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      col_stddevs(j) += std::pow(X(i, j) - col_means(j), 2);
    }
    col_stddevs(j) = std::sqrt(col_stddevs(j) / (n - 1));
  }

  // Standardize the matrix
  for (int j = 0; j < q; j++) {
    for (int i = 0; i < n; i++) {
      X_std(i, j) = (X(i, j) - col_means(j)) / col_stddevs(j);
    }
  }

  return X_std;
}
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive (VAR) Model
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param time Integer.
//'   Number of time points to simulate.
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results.
//' @param constant Numeric vector.
//'   The constant term vector of length `k`,
//'   where `k` is the number of variables.
//' @param coef Numeric matrix.
//'   Coefficient matrix with dimensions `k` by `(k * p)`.
//'   Each `k` by `k` block corresponds to the coefficient matrix
//'   for a particular lag.
//' @param chol_cov Numeric matrix.
//'   The Cholesky decomposition of the covariance matrix
//'   of the multivariate normal noise.
//'   It should have dimensions `k` by `k`.
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
//'
//' @examples
//' set.seed(42)
//' time <- 50L
//' burn_in <- 10L
//' k <- 3
//' p <- 2
//' constant <- c(1, 1, 1)
//' coef <- matrix(
//'   data = c(
//'     0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
//'     0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
//'     0.0, 0.0, 0.6, 0.0, 0.0, 0.3
//'   ),
//'   nrow = k,
//'   byrow = TRUE
//' )
//' chol_cov <- chol(diag(3))
//' y <- SimVAR(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov
//' )
//' head(y)
//'
//' @details
//' The [SimVAR()] function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model.
//' The VAR model is defined by the constant term `constant`,
//' the coefficient matrix `coef`,
//' and the Cholesky decomposition of the covariance matrix
//' of the multivariate normal process noise `chol_cov`.
//' The generated time series data follows a VAR(p) process,
//' where `p` is the number of lags specified by the size of `coef`.
//' The generated data includes a burn-in period,
//' which is excluded before returning the results.
//'
//' The steps involved in generating the VAR time series data are as follows:
//'
//' - Extract the number of variables `k` and the number of lags `p`
//'   from the input.
//' - Create a matrix `data` of size `k` by (`time + burn_in`)
//'   to store the generated VAR time series data.
//' - Set the initial values of the matrix `data`
//'   using the constant term `constant`.
//' - For each time point starting from the `p`-th time point
//'   to `time + burn_in - 1`:
//'   * Generate a vector of random noise
//'     from a multivariate normal distribution
//'     with mean 0 and covariance matrix `chol_cov`.
//'   * Generate the VAR time series values for each variable `j` at time `t`
//'     using the formula:
//'     \deqn{Y_{tj} = constant_j +
//'     \sum_{l = 1}^{p} \sum_{m = 1}^{k} (coef_{jm} * Y_{im}) +
//'     \text{noise}_{j}}
//'     where \eqn{Y_{tj}} is the value of variable `j` at time `t`,
//'     \eqn{constant_j} is the constant term for variable `j`,
//'     \eqn{coef_{jm}} are the coefficients for variable `j`
//'     from lagged variables up to order `p`,
//'     \eqn{Y_{tm}} are the lagged values of variable `m`
//'     up to order `p` at time `t`,
//'     and \eqn{noise_{j}} is the element `j`
//'     from the generated vector of random process noise.
//' - Transpose the matrix `data` and return only
//'   the required time period after the burn-in period,
//'   which is from column `burn_in` to column `time + burn_in - 1`.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov) {
  int k = constant.n_elem;  // Number of variables
  int p = coef.n_cols / k;  // Order of the VAR model (number of lags)

  int total_time = burn_in + time;

  // Matrix to store the generated VAR time series data
  arma::mat data(k, total_time);

  // Set initial values using the constant term
  data.each_col() = constant;  // Fill each column with the constant vector

  // Generate the VAR time series
  for (int t = p; t < total_time; t++) {
    // Generate noise from a multivariate normal distribution
    arma::vec noise = arma::randn(k);
    arma::vec mult_noise = chol_cov * noise;

    // Generate eta_t vector
    for (int j = 0; j < k; j++) {
      for (int lag = 0; lag < p; lag++) {
        for (int l = 0; l < k; l++) {
          data(j, t) += coef(j, lag * k + l) * data(l, t - lag - 1);
        }
      }

      data(j, t) += mult_noise(j);
    }
  }

  // Remove the burn-in period
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Create Y and X Matrices
//'
//' This function creates the dependent variable (Y)
//' and predictor variable (X) matrices.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param data Numeric matrix.
//'   The time series data with dimensions `t` by `k`,
//'   where `t` is the number of observations
//'   and `k` is the number of variables.
//' @param p Integer.
//'   The order of the VAR model (number of lags).
//'
//' @return List containing the dependent variable (Y)
//' and predictor variable (X) matrices.
//' Note that the resulting matrices will have `t - p` rows.
//'
//' @examples
//' set.seed(42)
//' time <- 50L
//' burn_in <- 10L
//' k <- 3
//' p <- 2
//' constant <- c(1, 1, 1)
//' coef <- matrix(
//'   data = c(
//'     0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
//'     0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
//'     0.0, 0.0, 0.6, 0.0, 0.0, 0.3
//'   ),
//'   nrow = k,
//'   byrow = TRUE
//' )
//' chol_cov <- chol(diag(3))
//' y <- SimVAR(
//'   time = time,
//'   burn_in = burn_in,
//'   constant = constant,
//'   coef = coef,
//'   chol_cov = chol_cov
//' )
//' yx <- YX(data = y, p = 2)
//' str(yx)
//'
//' @details
//' The [YX()] function creates the `Y` and `X` matrices
//' required for fitting a Vector Autoregressive (VAR) model.
//' Given the input `data` matrix with dimensions `t` by `k`,
//' where `t` is the number of observations and `k` is the number of variables,
//' and the order of the VAR model `p` (number of lags),
//' the function constructs lagged predictor matrix `X`
//' and the dependent variable matrix `Y`.
//'
//' The steps involved in creating the `Y` and `X` matrices are as follows:
//'
//' - Determine the number of observations `t` and the number of variables `k`
//'   from the input data matrix.
//' - Create matrices `X` and `Y` to store lagged variables
//'   and the dependent variable, respectively.
//' - Populate the matrices `X` and `Y` with the appropriate lagged data.
//'   The predictors matrix `X` contains a column of ones
//'   and the lagged values of the dependent variables,
//'   while the dependent variable matrix `Y` contains the original values
//'   of the dependent variables.
//' - The function returns a list containing the `Y` and `X` matrices,
//'   which can be used for further analysis and estimation
//'   of the VAR model parameters.
//'
//' @seealso
//' The [SimVAR()] function for simulating time series data
//' from a VAR model.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg utils
//' @export
// [[Rcpp::export]]
Rcpp::List YX(const arma::mat& data, int p) {
  int t = data.n_rows;  // Number of observations
  int k = data.n_cols;  // Number of variables

  // Create matrices to store lagged variables and the dependent variable
  arma::mat X(t - p, k * p + 1, arma::fill::zeros);  // Add 1 column for the
                                                     // constant
  arma::mat Y(t - p, k, arma::fill::zeros);

  // Populate the matrices X and Y with lagged data
  for (int i = 0; i < (t - p); i++) {
    X(i, 0) = 1;  // Set the first column to 1 for the constant term
    int index = 1;
    // Arrange predictors from smallest lag to biggest
    for (int lag = p - 1; lag >= 0; lag--) {
      X.row(i).subvec(index, index + k - 1) = data.row(i + lag);
      index += k;
    }
    Y.row(i) = data.row(i + p);
  }

  // Create a list to store X, Y
  Rcpp::List result;
  result["X"] = X;
  result["Y"] = Y;

  return result;
}

// Dependencies
// simAutoReg-sim-var.cpp
