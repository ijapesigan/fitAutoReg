// -----------------------------------------------------------------------------
// edit .setup/cpp/000-forward-declarations.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov);

arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov,
                    const arma::mat& exo_mat, const arma::mat& exo_coef);

Rcpp::List YX(const arma::mat& data, int p);

Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat);

arma::mat OrigScale(const arma::mat& coef_std, const arma::mat& Y,
                    const arma::mat& X);

arma::mat StdMat(const arma::mat& X);

arma::mat FitVAROLS(const arma::mat& Y, const arma::mat& X);

arma::vec LambdaSeq(const arma::mat& YStd, const arma::mat& XStd,
                    int n_lambdas);

arma::mat FitVARLasso(const arma::mat& YStd, const arma::mat& XStd,
                      const double& lambda, int max_iter, double tol);

arma::mat FitVARLassoSearch(const arma::mat& YStd, const arma::mat& XStd,
                            const arma::vec& lambdas, const std::string& crit,
                            int max_iter, double tol);

Rcpp::List SearchVARLasso(const arma::mat& YStd, const arma::mat& XStd,
                          const arma::vec& lambdas, int max_iter, double tol);

arma::vec PBootVAROLSRep(int time, int burn_in, const arma::vec& constant,
                         const arma::mat& coef, const arma::mat& chol_cov);

arma::mat PBootVAROLSSim(int B, int time, int burn_in,
                         const arma::vec& constant, const arma::mat& coef,
                         const arma::mat& chol_cov);

Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B, int burn_in);

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

Rcpp::List RBootVAROLS(const arma::mat& data, int p, int B);

Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B, int n_lambdas,
                         const std::string& crit, int max_iter, double tol);

Rcpp::List RBootVARExoOLS(const arma::mat& data, const arma::mat& exo_mat,
                          int p, int B);

Rcpp::List RBootVARExoLasso(const arma::mat& data, const arma::mat& exo_mat,
                            int p, int B, int n_lambdas,
                            const std::string& crit, int max_iter, double tol);

arma::vec PBootVARExoOLSRep(int time, int burn_in, const arma::vec& constant,
                            const arma::mat& coef, const arma::mat& chol_cov,
                            const arma::mat& exo_mat,
                            const arma::mat& exo_coef);

arma::mat PBootVARExoOLSSim(int B, int time, int burn_in,
                            const arma::vec& constant, const arma::mat& coef,
                            const arma::mat& chol_cov, const arma::mat& exo_mat,
                            const arma::mat& exo_coef);

Rcpp::List PBootVARExoOLS(const arma::mat& data, const arma::mat& exo_mat,
                          int p, int B, int burn_in);
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso-search.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters
//' using Lasso Regularization with Lambda Search
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param lambdas Numeric vector.
//'   Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param crit Character string.
//'   Information criteria to use.
//'   Valid values include `"aic"`, `"bic"`, and `"ebic"`.
//' @inheritParams FitVARLasso
//'
//' @return Matrix of estimated autoregressive
//'   and cross-regression coefficients.
//'
//' @examples
//' YStd <- StdMat(dat_p2_yx$Y)
//' XStd <- StdMat(dat_p2_yx$X[, -1]) # remove the constant column
//' lambdas <- LambdaSeq(
//'   YStd = YStd,
//'   XStd = XStd,
//'   n_lambdas = 100
//' )
//' FitVARLassoSearch(
//'   YStd = YStd,
//'   XStd = XStd,
//'   lambdas = lambdas,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLassoSearch(const arma::mat& YStd, const arma::mat& XStd,
                            const arma::vec& lambdas, const std::string& crit,
                            int max_iter, double tol) {
  // Step 1: Get the number of time points and predictor variables
  int time = XStd.n_rows;
  int num_predictor_vars = XStd.n_cols;

  // Step 2: Initialize variables to keep track of the best model based on the
  // selected criterion
  double min_criterion = std::numeric_limits<double>::infinity();
  arma::mat coef_min_crit;

  // Step 3: Loop over each lambda value in the 'lambdas' vector
  for (arma::uword i = 0; i < lambdas.n_elem; ++i) {
    double lambda = lambdas(i);

    // Step 4: Fit a VAR model with Lasso regularization for the current lambda
    // value
    arma::mat coef = FitVARLasso(YStd, XStd, lambda, max_iter, tol);

    // Step 5: Calculate the residuals, RSS, and the number of nonzero
    // parameters
    arma::mat residuals = YStd - XStd * coef.t();
    double rss = arma::accu(residuals % residuals);
    int num_params = arma::sum(arma::vectorise(coef != 0));

    // Step 6: Calculate the chosen information criterion (AIC, BIC, or EBIC)
    double aic = time * std::log(rss / time) + 2.0 * num_params;
    double bic = time * std::log(rss / time) + num_params * std::log(time);
    double ebic =
        time * std::log(rss / time) +
        2.0 * num_params * std::log(time / double(num_predictor_vars));

    // Step 7: Determine the current criterion value based on user choice
    // ('aic', 'bic', or 'ebic')
    double current_criterion = 0.0;
    if (crit == "aic") {
      current_criterion = aic;
    } else if (crit == "bic") {
      current_criterion = bic;
    } else if (crit == "ebic") {
      current_criterion = ebic;
    }

    // Step 8: Update the best model if the current criterion is smaller than
    // the minimum
    if (current_criterion < min_criterion) {
      min_criterion = current_criterion;
      coef_min_crit = coef;
    }
  }

  // Step 9: Return the coefficients of the best model based on the selected
  // criterion
  return coef_min_crit;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters
//' using Lasso Regularization
//'
//' This function estimates the parameters of a VAR model
//' using the Lasso regularization method
//' with cyclical coordinate descent.
//' The Lasso method is used to estimate the autoregressive
//' and cross-regression coefficients with sparsity.
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @param YStd Numeric matrix.
//'   Matrix of standardized dependent variables (Y).
//' @param XStd Numeric matrix.
//'   Matrix of standardized predictors (X).
//'   `XStd` should not include a vector of ones in column one.
//' @param lambda Numeric.
//'   Lasso hyperparameter.
//'   The regularization strength controlling the sparsity.
//' @param max_iter Integer.
//'   The maximum number of iterations
//'   for the coordinate descent algorithm
//'   (e.g., `max_iter = 10000`).
//' @param tol Numeric.
//'   Convergence tolerance.
//'   The algorithm stops when the change in coefficients
//'   between iterations is below this tolerance
//'   (e.g., `tol = 1e-5`).
//'
//' @return Matrix of estimated autoregressive
//'   and cross-regression coefficients.
//'
//' @examples
//' YStd <- StdMat(dat_p2_yx$Y)
//' XStd <- StdMat(dat_p2_yx$X[, -1]) # remove the constant column
//' lambda <- 73.90722
//' FitVARLasso(
//'   YStd = YStd,
//'   XStd = XStd,
//'   lambda = lambda,
//'   max_iter = 10000,
//'   tol = 1e-5
//' )
//'
//' @details
//' The [fitAutoReg::FitVARLasso()] function estimates the parameters
//' of a Vector Autoregressive (VAR) model
//' using the Lasso regularization method.
//' Given the input matrices `YStd` and `XStd`,
//' where `YStd` is the matrix of standardized dependent variables,
//' and `XStd` is the matrix of standardized predictors,
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
//'   (columns of `XStd`),
//'   while the inner loop iterates over the outcome variables
//'   (columns of `YStd`).
//' - **Coefficient Update**: For each predictor variable (column of `XStd`),
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
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVARLasso(const arma::mat& YStd, const arma::mat& XStd,
                      const double& lambda, int max_iter, double tol) {
  // Step 1: Determine the number of predictor variables and outcome variables
  int num_predictor_vars = XStd.n_cols;
  int num_outcome_vars = YStd.n_cols;

  // Step 2: Initialize matrices to store QR decomposition results
  arma::mat Q, R;

  // Step 3: Perform QR decomposition of the standardized predictor matrix XStd
  arma::qr_econ(Q, R, XStd);

  // Step 4: Initialize a matrix 'coef' to store the estimated coefficients
  arma::mat coef = arma::solve(R, Q.t() * YStd);

  // Step 5: Iterate for a maximum of 'max_iter' times
  for (int iter = 0; iter < max_iter; iter++) {
    // Step 5.1: Create a copy of the current 'coef' matrix for comparison
    arma::mat coef_old = coef;

    // Step 5.2: Create a copy of the outcome matrix 'YStd' for updates
    arma::mat Y_copy = YStd;

    // Step 5.3: Loop over predictor variables
    for (int j = 0; j < num_predictor_vars; j++) {
      // Step 5.3.1: Extract the j-th column of the standardized predictor
      // matrix XStd
      arma::vec Xj = XStd.col(j);

      // Step 5.3.2: Loop over outcome variables
      for (int l = 0; l < num_outcome_vars; l++) {
        // Step 5.3.2.1: Extract the l-th column of the copy of the outcome
        // matrix Y_copy
        arma::vec Y_l = Y_copy.col(l);

        // Step 5.3.2.2: Compute 'rho' and 'z' for Lasso regularization
        double rho = dot(Xj, Y_l - XStd * coef.col(l) + coef(j, l) * Xj);
        double z = dot(Xj, Xj);

        // Step 5.3.2.3: Apply Lasso regularization and update the 'coef' matrix
        double c = 0;
        if (rho < -lambda / 2) {
          c = (rho + lambda / 2) / z;
        } else if (rho > lambda / 2) {
          c = (rho - lambda / 2) / z;
        } else {
          c = 0;
        }
        coef(j, l) = c;

        // Step 5.3.2.4: Update the l-th column of the copy of the outcome
        // matrix Y_copy
        Y_l = Y_l - (Xj * (coef(j, l) - coef_old(j, l)));
      }
    }

    // Step 5.4: Check for convergence based on the change in 'coef'
    if (iter > 0) {
      if (arma::all(arma::vectorise(arma::abs(coef - coef_old)) < tol)) {
        break;
      }
    }

    // Step 5.5: If maximum iterations are reached without convergence, issue a
    // warning
    if (iter == max_iter - 1) {
      Rcpp::warning(
          "The algorithm did not converge within the specified maximum number "
          "of iterations.");
    }
  }

  // Step 6: Return the estimated coefficients (transposed for the desired
  // format)
  return coef.t();
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-fit-var-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Fit Vector Autoregressive (VAR) Model Parameters
//' using Ordinary Least Squares (OLS)
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
//'   and cross-regression coefficients.
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
//' the function computes the autoregressive
//' and cross-regression coefficients of the VAR model.
//' Note that if the first column of `X` is a vector of ones,
//' the constant vector is also estimated.
//'
//' The steps involved in estimating the VAR model parameters
//' using OLS are as follows:
//'
//' - Compute the QR decomposition of the lagged predictor matrix `X`
//'   using the `qr_econ` function from the Armadillo library.
//' - Extract the `Q` and `R` matrices from the QR decomposition.
//' - Solve the linear system `R * coef = Q.t() * Y`
//'   to estimate the VAR model coefficients `coef`.
//' - The function returns a matrix containing the estimated
//'   autoregressive and cross-regression coefficients of the VAR model.
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg fit
//' @export
// [[Rcpp::export]]
arma::mat FitVAROLS(const arma::mat& Y, const arma::mat& X) {
  // Step 1: Initialize matrices to store QR decomposition results
  arma::mat Q, R;

  // Step 2: Perform QR decomposition of the design matrix X
  arma::qr_econ(Q, R, X);

  // Step 3: Solve the linear system to obtain the coefficient matrix
  //   - Transpose of Q (Q.t()) is multiplied by Y to project Y
  //     onto the column space of X
  //   - R is the upper triangular matrix from the QR decomposition
  //   - arma::solve(R, ...) solves the linear system
  //     R * coef = Q.t() * Y for coef
  arma::mat coef = arma::solve(R, Q.t() * Y);

  // Step 4: Transpose the coefficient matrix to match the desired output format
  //         (columns represent variables, rows represent lags)
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
//' X <- dat_p2_yx$X[, -1] # remove the constant column
//' YStd <- StdMat(Y)
//' XStd <- StdMat(X)
//' coef_std <- FitVAROLS(Y = YStd, X = XStd)
//' FitVAROLS(Y = Y, X = X)
//' OrigScale(coef_std = coef_std, Y = Y, X = X)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg utils
//' @export
// [[Rcpp::export]]
arma::mat OrigScale(const arma::mat& coef_std, const arma::mat& Y,
                    const arma::mat& X) {
  // Step 1: Get the number of outcome variables and predictor variables
  int num_outcome_vars = coef_std.n_rows;
  int num_predictor_vars = coef_std.n_cols;

  // Step 2: Initialize vectors to store standard deviations of outcome (Y) and
  // predictor (X) variables
  arma::vec sd_Y(num_outcome_vars);
  arma::vec sd_X(num_predictor_vars);

  // Step 3: Calculate the standard deviation for each outcome variable (Y)
  for (int l = 0; l < num_outcome_vars; l++) {
    sd_Y(l) = arma::as_scalar(arma::stddev(Y.col(l), 0, 0));
  }

  // Step 4: Calculate the standard deviation for each predictor variable (X)
  for (int j = 0; j < num_predictor_vars; j++) {
    sd_X(j) = arma::as_scalar(arma::stddev(X.col(j), 0, 0));
  }

  // Step 5: Initialize a matrix 'orig' to store coefficients in the original
  // scale
  arma::mat orig(num_outcome_vars, num_predictor_vars);

  // Step 6: Compute original-scale coefficients by scaling back from
  // standardized coefficients
  for (int l = 0; l < num_outcome_vars; l++) {
    for (int j = 0; j < num_predictor_vars; j++) {
      double orig_coef = coef_std(l, j) * sd_Y(l) / sd_X(j);
      orig(l, j) = orig_coef;
    }
  }

  // Step 7: Return the coefficients in the original scale
  return orig;
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
  // Step 1: Determine the number of lags in the VAR model
  int num_lags = coef.n_cols / constant.n_elem;

  // Step 2: Simulate data from VAR model with given parameters
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // Step 3: Convert the data into YX representation with lagged variables
  Rcpp::List yx = YX(data, num_lags);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 4: Exclude the constant column from X for standardization
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);

  // Step 5: Fit an OLS model to estimate the constant vector
  arma::mat ols = FitVAROLS(Y, X);
  arma::vec const_b = ols.col(0);

  // Step 6: Standardize the predictor and response variables
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 7: Generate a sequence of lambda values for LASSO regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 8: Fit VAR LASSO model to the standardized data
  arma::mat coef_std_b =
      FitVARLassoSearch(YStd, XStd, lambdas, crit, max_iter, tol);

  // Step 9: Rescale the estimated coefficients to their original scale
  arma::mat coef_orig = OrigScale(coef_std_b, Y, X_no_constant);

  // Step 10: Combine the estimated constant and original scale coefficients
  arma::mat coef_b = arma::join_horiz(const_b, coef_orig);

  // Step 11: Vectorize the coefficient matrix for bootstrapping
  return arma::vectorise(coef_b);
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
  // Step 1: Determine the number of coefficients (constant + VAR coefficients)
  int num_coef = constant.n_elem + coef.n_elem;

  // Step 2: Initialize a matrix to store bootstrap results
  arma::mat result(B, num_coef, arma::fill::zeros);

  // Step 3: Bootstrap the VAR LASSO coefficients B times
  for (int b = 0; b < B; b++) {
    // Step 4: Call PBootVARLassoRep to perform bootstrap replication
    arma::vec coef_b = PBootVARLassoRep(time, burn_in, constant, coef, chol_cov,
                                        n_lambdas, crit, max_iter, tol);

    // Step 5: Store the bootstrap results in the result matrix
    result.row(b) = arma::trans(coef_b);
  }

  // Step 6: Return the matrix containing bootstrap results
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
//' @inheritParams PBootVAROLS
//' @inheritParams LambdaSeq
//' @inheritParams FitVARLassoSearch
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original Lasso estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'
//' @examples
//' pb <- PBootVARLasso(
//'   data = dat_p2,
//'   p = 2,
//'   B = 5,
//'   burn_in = 20,
//'   n_lambdas = 10,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVARLasso(const arma::mat& data, int p, int B, int burn_in,
                         int n_lambdas, const std::string& crit, int max_iter,
                         double tol) {
  // Step 1: Get the number of time periods
  int time = data.n_rows;

  // Step 2: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 3: Remove the constant term from the lagged variables
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);

  // Step 4: Fit a VAR model using OLS to obtain the constant term and VAR
  // coefficients
  arma::mat ols = FitVAROLS(Y, X);

  // Step 5: Standardize the predictor and response variables
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 6: Generate a sequence of lambda values for LASSO regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 7: Fit VAR LASSO using the "ebic" criterion
  arma::mat pb_std = FitVARLassoSearch(YStd, XStd, lambdas, "ebic", 1000, 1e-5);

  // Step 8: Extract the constant term from the OLS results
  arma::vec const_vec = ols.col(0);

  // Step 9: Rescale the VAR LASSO coefficients to the original scale
  arma::mat coef_mat = OrigScale(pb_std, Y, X_no_constant);

  // Step 10: Combine the constant and VAR coefficients
  arma::mat coef =
      arma::join_horiz(const_vec, coef_mat);  // OLS and Lasso combined

  // Step 11: Calculate residuals, their covariance, and the Cholesky
  // decomposition of the covariance
  arma::mat residuals = Y - X * coef.t();
  arma::mat cov_residuals = arma::cov(residuals);
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Step 12: Perform bootstrap simulations for VAR LASSO
  arma::mat sim = PBootVARLassoSim(B, time, burn_in, const_vec, coef_mat,
                                   chol_cov, n_lambdas, crit, max_iter, tol);

  // Step 13: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = sim;

  // Step 14: Return the list of results
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
  // Step 1: Calculate the number of lags in the VAR model
  int num_lags = coef.n_cols / constant.n_elem;

  // Step 2: Simulate a VAR process using the SimVAR function
  arma::mat data = SimVAR(time, burn_in, constant, coef, chol_cov);

  // Step 3: Create lagged matrices X and Y using the YX function
  Rcpp::List yx = YX(data, num_lags);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 4: Estimate VAR coefficients using the FitVAROLS function
  arma::mat coef_b = FitVAROLS(Y, X);

  // Step 5: Return the estimated coefficients as a vector
  return arma::vectorise(coef_b);
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
  // Step 1: Calculate the total number of coefficients in the VAR model
  int num_coef = constant.n_elem + coef.n_elem;

  // Step 2: Initialize the result matrix to store bootstrapped coefficient
  // estimates
  arma::mat result(B, num_coef, arma::fill::zeros);

  // Step 3: Perform bootstrapping B times
  for (int b = 0; b < B; b++) {
    // Step 4: Obtain bootstrapped VAR coefficient estimates using
    // PBootVAROLSRep
    arma::vec coef_est =
        PBootVAROLSRep(time, burn_in, constant, coef, chol_cov);

    // Step 5: Store the estimated coefficients in the result matrix
    result.row(b) = arma::trans(coef_est);
  }

  // Step 6: Return the result matrix containing bootstrapped coefficient
  // estimates
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
//' @inheritParams RBootVAROLS
//' @param burn_in Integer.
//'   Number of burn-in observations to exclude before returning the results
//'   in the simulation step.
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original OLS estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'
//' @examples
//' pb <- PBootVAROLS(data = dat_p2, p = 2, B = 5, burn_in = 20)
//' str(pb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg pb
//' @export
// [[Rcpp::export]]
Rcpp::List PBootVAROLS(const arma::mat& data, int p, int B, int burn_in) {
  // Step 1: Get the number of time points in the data
  int time = data.n_rows;

  // Step 2: Obtain the YX representation of the data
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];

  // Step 3: Fit the VAR model using OLS
  arma::mat coef = FitVAROLS(Y, X);

  // Step 4: Extract constant vector and coefficient matrix
  arma::vec const_vec = coef.col(0);
  arma::mat coef_mat = coef.cols(1, coef.n_cols - 1);

  // Step 5: Calculate residuals and their covariance
  arma::mat residuals = Y - X * coef.t();
  arma::mat cov_residuals = arma::cov(residuals);

  // Step 6: Perform Cholesky decomposition of the covariance matrix
  arma::mat chol_cov = arma::chol(cov_residuals);

  // Step 7: Simulate bootstrapped VAR coefficients using PBootVAROLSSim
  arma::mat sim =
      PBootVAROLSSim(B, time, burn_in, const_vec, coef_mat, chol_cov);

  // Step 8: Create a result list containing estimated coefficients and
  // bootstrapped samples
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = sim;

  // Step 9: Return the result list
  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-r-boot-var-exo-lasso.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Residual Bootstrap for the Vector Autoregressive Model
//' with Exogenous Variables
//' Using Lasso Regularization
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @inheritParams RBootVARLasso
//' @inheritParams RBootVARExoOLS
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original Lasso estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'   - **X**: Numeric matrix.
//'     Original `X`
//'   - **Y**: List of numeric matrices.
//'     Bootstrapped `Y`
//'
//' @examples
//' data <- dat_p2_exo$data
//' exo_mat <- dat_p2_exo$exo_mat
//' rb <- RBootVARExoLasso(
//'   data = data,
//'   exo_mat = exo_mat,
//'   p = 2,
//'   B = 5,
//'   n_lambdas = 10,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARExoLasso(const arma::mat& data, const arma::mat& exo_mat,
                            int p, int B, int n_lambdas,
                            const std::string& crit, int max_iter, double tol) {
  // Step 1: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YXExo(data, p, exo_mat);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);
  int time = Y.n_rows;

  // Step 2: Fit a VAR model using OLS to obtain the VAR coefficients
  arma::mat ols = FitVAROLS(Y, X);

  // Step 3: Standardize the data
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 4: Generate a sequence of lambda values for Lasso regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 5: Fit VAR Lasso to obtain standardized coefficients
  arma::mat coef_std =
      FitVARLassoSearch(YStd, XStd, lambdas, "ebic", max_iter, tol);

  // Step 6: Extract the constant vector from OLS results
  arma::vec const_vec = ols.col(0);

  // Step 7: Transform standardized coefficients back to the original scale
  arma::mat coef_mat = OrigScale(coef_std, Y, X_no_constant);

  // Step 8: Combine the constant and coefficient matrices
  arma::mat coef = arma::join_horiz(const_vec, coef_mat);

  // Step 9: Calculate residuals based on the estimated VAR coefficients
  arma::mat residuals = Y - X * coef.t();

  // Step 10: Prepare containers for bootstrap results
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);
  Rcpp::List Y_list(B);

  // Step 11: Perform B bootstrap simulations
  for (int b = 0; b < B; ++b) {
    // 11.1: Randomly select rows from residuals to create a new residuals
    // matrix for the bootstrap sample
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // 11.2: Generate a new response matrix Y_b by adding the new residuals to X
    // * coef.t()
    arma::mat Y_b = X * coef.t() + residuals_b;

    // 11.3: Fit a VAR model using OLS to obtain VAR coefficients for the
    // bootstrap sample
    arma::mat ols_b = FitVAROLS(Y_b, X);

    // 11.4: Standardize the Y_b matrix
    arma::mat YStd_b = StdMat(Y);

    // 11.5: Fit VAR Lasso to obtain standardized coefficients for the bootstrap
    // sample
    arma::mat coef_std_b =
        FitVARLassoSearch(YStd_b, XStd, lambdas, "ebic", max_iter, tol);

    // 11.6: Extract the constant vector from OLS results for the bootstrap
    // sample
    arma::vec const_vec_b = ols_b.col(0);

    // 11.7: Transform standardized coefficients back to the original scale for
    // the bootstrap sample
    arma::mat coef_mat_b = OrigScale(coef_std_b, Y_b, X_no_constant);

    // 11.8: Combine the constant and coefficient matrices for the bootstrap
    // sample
    arma::mat coef_lasso_b = arma::join_horiz(const_vec_b, coef_mat_b);

    // 11.9: Vectorize the coefficients and store them in coef_b_mat
    arma::vec coef_b = arma::vectorise(coef_lasso_b);
    coef_b_mat.col(b) = coef_b;

    // 11.10: Store the Y_b matrix as an Rcpp list element
    Y_list[b] = Rcpp::wrap(Y_b);
  }

  // Step 12: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = coef_b_mat.t();
  result["X"] = X;
  result["Y"] = Y_list;

  // Step 13: Return the list of results
  return result;
}
// -----------------------------------------------------------------------------
// edit .setup/cpp/fitAutoReg-r-boot-var-exo-ols.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Residual Bootstrap for the Vector Autoregressive Model
//' with Exogenous Variables
//' Using Ordinary Least Squares
//'
//' @author Ivan Jacob Agaloos Pesigan
//'
//' @inheritParams RBootVAROLS
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous variables with dimensions `t` by `m`.
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original OLS estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'   - **X**: Numeric matrix.
//'     Original `X`
//'   - **Y**: List of numeric matrices.
//'     Bootstrapped `Y`
//'
//' @examples
//' data <- dat_p2_exo$data
//' exo_mat <- dat_p2_exo$exo_mat
//' rb <- RBootVARExoOLS(data = data, exo_mat = exo_mat, p = 2, B = 5)
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARExoOLS(const arma::mat& data, const arma::mat& exo_mat,
                          int p, int B) {
  // Step 1: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YXExo(data, p, exo_mat);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  int time = Y.n_rows;

  // Step 2: Fit a VAR model using OLS to obtain the VAR coefficients
  arma::mat coef = FitVAROLS(Y, X);

  // Step 3: Calculate residuals based on the estimated VAR coefficients
  arma::mat residuals = Y - X * coef.t();

  // Step 4: Prepare containers for bootstrap results
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);
  Rcpp::List Y_b_list(B);

  // Step 5: Perform B bootstrap simulations
  for (int b = 0; b < B; ++b) {
    // 5.1: Randomly select rows from residuals to create a new residuals matrix
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // 5.2: Generate a new response matrix Y_b by adding the new residuals to X
    // * coef.t()
    arma::mat Y_b = X * coef.t() + residuals_b;

    // 5.3: Fit a VAR model using OLS to obtain VAR coefficients for the
    // bootstrap sample
    arma::mat coef_ols_b = FitVAROLS(Y_b, X);

    // 5.4: Vectorize the coefficients and store them in coef_b_mat
    arma::vec coef_b = arma::vectorise(coef_ols_b);
    coef_b_mat.col(b) = coef_b;

    // 5.5: Store the Y_b matrix as an Rcpp list element
    Y_b_list[b] = Rcpp::wrap(Y_b);
  }

  // Step 6: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = coef_b_mat.t();
  result["X"] = X;
  result["Y"] = Y_b_list;

  // Step 7: Return the list of results
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
//' @inheritParams RBootVAROLS
//' @inheritParams FitVARLassoSearch
//' @inheritParams LambdaSeq
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original Lasso estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'   - **X**: Numeric matrix.
//'     Original `X`
//'   - **Y**: List of numeric matrices.
//'     Bootstrapped `Y`
//'
//' @examples
//' rb <- RBootVARLasso(
//'   data = dat_p2,
//'   p = 2,
//'   B = 5,
//'   n_lambdas = 10,
//'   crit = "ebic",
//'   max_iter = 1000,
//'   tol = 1e-5
//' )
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVARLasso(const arma::mat& data, int p, int B, int n_lambdas,
                         const std::string& crit, int max_iter, double tol) {
  // Step 1: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  arma::mat X_no_constant = X.cols(1, X.n_cols - 1);
  int time = Y.n_rows;

  // Step 2: Fit a VAR model using OLS to obtain the VAR coefficients
  arma::mat ols = FitVAROLS(Y, X);

  // Step 3: Standardize the data
  arma::mat XStd = StdMat(X_no_constant);
  arma::mat YStd = StdMat(Y);

  // Step 4: Generate a sequence of lambda values for Lasso regularization
  arma::vec lambdas = LambdaSeq(YStd, XStd, n_lambdas);

  // Step 5: Fit VAR Lasso to obtain standardized coefficients
  arma::mat coef_std =
      FitVARLassoSearch(YStd, XStd, lambdas, "ebic", max_iter, tol);

  // Step 6: Extract the constant vector from OLS results
  arma::vec const_vec = ols.col(0);

  // Step 7: Transform standardized coefficients back to the original scale
  arma::mat coef_mat = OrigScale(coef_std, Y, X_no_constant);

  // Step 8: Combine the constant and coefficient matrices
  arma::mat coef = arma::join_horiz(const_vec, coef_mat);

  // Step 9: Calculate residuals based on the estimated VAR coefficients
  arma::mat residuals = Y - X * coef.t();

  // Step 10: Prepare containers for bootstrap results
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);
  Rcpp::List Y_list(B);

  // Step 11: Perform B bootstrap simulations
  for (int b = 0; b < B; ++b) {
    // 11.1: Randomly select rows from residuals to create a new residuals
    // matrix for the bootstrap sample
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // 11.2: Generate a new response matrix Y_b by adding the new residuals to X
    // * coef.t()
    arma::mat Y_b = X * coef.t() + residuals_b;

    // 11.3: Fit a VAR model using OLS to obtain VAR coefficients for the
    // bootstrap sample
    arma::mat ols_b = FitVAROLS(Y_b, X);

    // 11.4: Standardize the Y_b matrix
    arma::mat YStd_b = StdMat(Y);

    // 11.5: Fit VAR Lasso to obtain standardized coefficients for the bootstrap
    // sample
    arma::mat coef_std_b =
        FitVARLassoSearch(YStd_b, XStd, lambdas, "ebic", max_iter, tol);

    // 11.6: Extract the constant vector from OLS results for the bootstrap
    // sample
    arma::vec const_vec_b = ols_b.col(0);

    // 11.7: Transform standardized coefficients back to the original scale for
    // the bootstrap sample
    arma::mat coef_mat_b = OrigScale(coef_std_b, Y_b, X_no_constant);

    // 11.8: Combine the constant and coefficient matrices for the bootstrap
    // sample
    arma::mat coef_lasso_b = arma::join_horiz(const_vec_b, coef_mat_b);

    // 11.9: Vectorize the coefficients and store them in coef_b_mat
    arma::vec coef_b = arma::vectorise(coef_lasso_b);
    coef_b_mat.col(b) = coef_b;

    // 11.10: Store the Y_b matrix as an Rcpp list element
    Y_list[b] = Rcpp::wrap(Y_b);
  }

  // Step 12: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = coef_b_mat.t();
  result["X"] = X;
  result["Y"] = Y_list;

  // Step 13: Return the list of results
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
//' @inheritParams YX
//' @param B Integer.
//'   Number of bootstrap samples to generate.
//'
//' @return List with the following elements:
//'   - **est**: Numeric matrix.
//'     Original OLS estimate of the coefficient matrix.
//'   - **boot**: Numeric matrix.
//'     Matrix of vectorized bootstrap estimates of the coefficient matrix.
//'   - **X**: Numeric matrix.
//'     Original `X`
//'   - **Y**: List of numeric matrices.
//'     Bootstrapped `Y`
//'
//' @examples
//' rb <- RBootVAROLS(data = dat_p2, p = 2, B = 5)
//' str(rb)
//'
//' @family Fitting Autoregressive Model Functions
//' @keywords fitAutoReg rb
//' @export
// [[Rcpp::export]]
Rcpp::List RBootVAROLS(const arma::mat& data, int p, int B) {
  // Step 1: Prepare the data for analysis by extracting lagged variables
  Rcpp::List yx = YX(data, p);
  arma::mat X = yx["X"];
  arma::mat Y = yx["Y"];
  int time = Y.n_rows;

  // Step 2: Fit a VAR model using OLS to obtain the VAR coefficients
  arma::mat coef = FitVAROLS(Y, X);

  // Step 3: Calculate residuals based on the estimated VAR coefficients
  arma::mat residuals = Y - X * coef.t();

  // Step 4: Prepare containers for bootstrap results
  arma::mat coef_b_mat(coef.n_rows * coef.n_cols, B);
  Rcpp::List Y_b_list(B);

  // Step 5: Perform B bootstrap simulations
  for (int b = 0; b < B; ++b) {
    // 5.1: Randomly select rows from residuals to create a new residuals matrix
    arma::mat residuals_b = residuals.rows(
        arma::randi<arma::uvec>(time, arma::distr_param(0, time - 1)));

    // 5.2: Generate a new response matrix Y_b by adding the new residuals to X
    // * coef.t()
    arma::mat Y_b = X * coef.t() + residuals_b;

    // 5.3: Fit a VAR model using OLS to obtain VAR coefficients for the
    // bootstrap sample
    arma::mat coef_ols_b = FitVAROLS(Y_b, X);

    // 5.4: Vectorize the coefficients and store them in coef_b_mat
    arma::vec coef_b = arma::vectorise(coef_ols_b);
    coef_b_mat.col(b) = coef_b;

    // 5.5: Store the Y_b matrix as an Rcpp list element
    Y_b_list[b] = Rcpp::wrap(Y_b);
  }

  // Step 6: Create a list containing estimation and bootstrap results
  Rcpp::List result;
  // Estimated coefficients
  result["est"] = coef;
  // Bootstrapped coefficient samples
  result["boot"] = coef_b_mat.t();
  result["X"] = X;
  result["Y"] = Y_b_list;

  // Step 7: Return the list of results
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
  // Step 1: Get the number of rows (n) and columns (num_vars) in the input
  // matrix X
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

  // Step 4: Calculate column standard deviations and store them in the col_sds
  // vector
  arma::vec col_sds(num_vars, arma::fill::zeros);
  for (int j = 0; j < num_vars; j++) {
    for (int i = 0; i < n; i++) {
      col_sds(j) += std::pow(X(i, j) - col_means(j), 2);
    }
    col_sds(j) = std::sqrt(
        col_sds(j) / (n - 1));  // Calculate the standard deviation for column j
  }

  // Step 5: Standardize the matrix X by subtracting column means and dividing
  // by column standard deviations
  for (int j = 0; j < num_vars; j++) {
    for (int i = 0; i < n; i++) {
      XStd(i, j) = (X(i, j) - col_means(j)) /
                   col_sds(j);  // Standardize each element of X
    }
  }

  // Step 6: Return the standardized matrix XStd
  return XStd;
}
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-sim-var-exo.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Simulate Data from a Vector Autoregressive (VAR) Model with Exogenous
//' Variables
//'
//' This function generates synthetic time series data
//' from a Vector Autoregressive (VAR) model with exogenous variables.
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
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous covariates with dimensions `time + burn_in` by `x`.
//'   Each column corresponds to a different exogenous variable.
//' @param exo_coef Numeric vector.
//'   Coefficient matrix with dimensions `k` by `x`
//'   associated with the exogenous covariates.
//'
//' @return Numeric matrix containing the simulated time series data
//'   with dimensions `k` by `time`,
//'   where `k` is the number of variables and
//'   `time` is the number of observations.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVARExo(int time, int burn_in, const arma::vec& constant,
                    const arma::mat& coef, const arma::mat& chol_cov,
                    const arma::mat& exo_mat, const arma::mat& exo_coef) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values for each outcome
  // variable
  data.each_col() = constant;

  // Step 4: Transpose the exogenous matrix for efficient column access
  arma::mat exo_mat_t = exo_mat.t();

  // Step 5: Simulate VAR-Exo data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 5.1: Generate random noise vector
    arma::vec noise = arma::randn(num_outcome_vars);

    // Step 5.2: Multiply the noise vector by the Cholesky decomposition of the
    // covariance matrix
    arma::vec mult_noise = chol_cov * noise;

    // Step 5.3: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 5.4: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 5.5: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 5.6: Iterate over exogenous variables
      for (arma::uword x = 0; x < exo_mat_t.n_rows; x++) {
        // Update data with exogenous variables and their coefficients
        data(j, t) += exo_mat_t(x, t) * exo_coef(j, x);
      }

      // Step 5.7: Add the corresponding element from the noise vector
      data(j, t) += mult_noise(j);
    }
  }

  // Step 6: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 7: Return the transposed data matrix
  return data.t();
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
//'     \deqn{
//'       Y_{tj} = \mathrm{constant}_j +
//'       \sum_{l = 1}^{p} \sum_{m = 1}^{k} (\mathrm{coef}_{jm} * Y_{im}) +
//'       \mathrm{noise}_{j}
//'     }
//'     where \eqn{Y_{tj}} is the value of variable `j` at time `t`,
//'     \eqn{\mathrm{constant}_j} is the constant term for variable `j`,
//'     \eqn{\mathrm{coef}_{jm}} are the coefficients for variable `j`
//'     from lagged variables up to order `p`,
//'     \eqn{Y_{tm}} are the lagged values of variable `m`
//'     up to order `p` at time `t`,
//'     and \eqn{\mathrm{noise}_{j}} is the element `j`
//'     from the generated vector of random process noise.
//' - Transpose the matrix `data` and return only
//'   the required time period after the burn-in period,
//'   which is from column `burn_in` to column `time + burn_in - 1`.
//'
//' @family Simulation of Autoregressive Data Functions
//' @keywords simAutoReg sim data var
//' @export
// [[Rcpp::export]]
arma::mat SimVAR(int time, int burn_in, const arma::vec& constant,
                 const arma::mat& coef, const arma::mat& chol_cov) {
  // Step 1: Determine dimensions and total time
  // Number of outcome variables
  int num_outcome_vars = constant.n_elem;
  // Number of lags in the VAR model
  int num_lags = coef.n_cols / num_outcome_vars;
  // Total number of time steps
  int total_time = burn_in + time;

  // Step 2: Create a matrix to store simulated data
  arma::mat data(num_outcome_vars, total_time);

  // Step 3: Initialize the data matrix with constant values for each outcome
  // variable
  data.each_col() = constant;

  // Step 4: Simulate VAR data using a loop
  for (int t = num_lags; t < total_time; t++) {
    // Step 4.1: Generate random noise vector
    arma::vec noise = arma::randn(num_outcome_vars);

    // Step 4.2: Multiply the noise vector
    //           by the Cholesky decomposition of the covariance matrix
    arma::vec mult_noise = chol_cov * noise;

    // Step 4.3: Iterate over outcome variables
    for (int j = 0; j < num_outcome_vars; j++) {
      // Step 4.4: Iterate over lags
      for (int lag = 0; lag < num_lags; lag++) {
        // Step 4.5: Iterate over outcome variables again
        for (int l = 0; l < num_outcome_vars; l++) {
          // Update data by applying VAR coefficients and lagged data
          data(j, t) +=
              coef(j, lag * num_outcome_vars + l) * data(l, t - lag - 1);
        }
      }

      // Step 4.6: Add the corresponding element from the noise vector
      data(j, t) += mult_noise(j);
    }
  }

  // Step 5: If there is a burn-in period, remove it
  if (burn_in > 0) {
    data = data.cols(burn_in, total_time - 1);
  }

  // Step 6: Return the transposed data matrix
  return data.t();
}

// Dependencies
// -----------------------------------------------------------------------------
// edit simAutoReg/.setup/cpp/simAutoReg-y-x.cpp
// -----------------------------------------------------------------------------

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Create Y and X Matrices with Exogenous Variables
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
//' @param exo_mat Numeric matrix.
//'   Matrix of exogenous variables with dimensions `t` by `m`.
//'
//' @return List containing the dependent variable (Y)
//' and predictor variable (X) matrices.
//' Note that the resulting matrices will have `t - p` rows.
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
Rcpp::List YXExo(const arma::mat& data, int p, const arma::mat& exo_mat) {
  // Step 1: Calculate the dimensions of the 'data' and 'exo_mat' matrices
  // Number of time steps (rows) in 'data'
  int time = data.n_rows;
  // Number of outcome variables (columns) in 'data'
  int num_outcome_vars = data.n_cols;
  // Number of exogenous variables (columns) in 'exo_mat'
  int num_exo_vars = exo_mat.n_cols;

  // Step 2: Create matrices 'X' and 'Y'
  //         to store transformed data 'X' matrix with ones
  arma::mat X(time - p, num_outcome_vars * p + num_exo_vars + 1,
              arma::fill::ones);
  // 'Y' matrix with zeros
  arma::mat Y(time - p, num_outcome_vars, arma::fill::zeros);

  // Step 3: Loop through the data and populate 'X' and 'Y'
  for (int time_index = 0; time_index < (time - p); time_index++) {
    // Initialize the column index for 'X'
    int index = 1;

    // Nested loop to populate 'X' with lagged values
    for (int lag = p - 1; lag >= 0; lag--) {
      // Update 'X' by assigning a subvector of 'data' to a subvector of 'X'
      X.row(time_index).subvec(index, index + num_outcome_vars - 1) =
          data.row(time_index + lag);
      // Move to the next set of columns in 'X'
      index += num_outcome_vars;
    }

    // Update 'X' with the exogenous variables
    X.row(time_index).subvec(index, index + num_exo_vars - 1) =
        exo_mat.row(time_index + p);

    // Update 'Y' with the target values
    Y.row(time_index) = data.row(time_index + p);
  }

  // Step 4: Create an Rcpp List 'result' and assign 'Y' and 'X' matrices to it
  Rcpp::List result;
  result["Y"] = Y;
  result["X"] = X;

  // Step 5: Return the 'result' List containing the transformed data
  return result;
}
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
  // Step 1: Calculate the dimensions of the 'data' matrix
  // Number of time steps (rows)
  int time = data.n_rows;
  // Number of outcome variables (columns)
  int num_outcome_vars = data.n_cols;

  // Step 2: Create matrices 'X' and 'Y'
  //         to store transformed data 'X' matrix with ones
  arma::mat X(time - p, num_outcome_vars * p + 1, arma::fill::ones);
  // 'Y' matrix with zeros
  arma::mat Y(time - p, num_outcome_vars, arma::fill::zeros);

  // Step 3: Loop through the data and populate 'X' and 'Y'
  for (int time_index = 0; time_index < (time - p); time_index++) {
    // Initialize the column index for 'X'
    int index = 1;

    // Nested loop to populate 'X' with lagged values
    for (int lag = p - 1; lag >= 0; lag--) {
      // Update 'X' by assigning a subvector of 'data' to a subvector of 'X'
      X.row(time_index).subvec(index, index + num_outcome_vars - 1) =
          data.row(time_index + lag);
      // Move to the next set of columns in 'X'
      index += num_outcome_vars;
    }

    // Update 'Y' with the target values
    Y.row(time_index) = data.row(time_index + p);
  }

  // Step 4: Create an Rcpp List 'result' and assign 'X' and 'Y' matrices to it
  Rcpp::List result;
  result["X"] = X;
  result["Y"] = Y;

  // Step 5: Return the 'result' List containing the transformed data
  return result;
}

// Dependencies
// simAutoReg-sim-var.cpp
