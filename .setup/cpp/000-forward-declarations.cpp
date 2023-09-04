// -----------------------------------------------------------------------------
// edit .setup/cpp/000-forward-declarations.cpp
// Ivan Jacob Agaloos Pesigan
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
