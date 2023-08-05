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
