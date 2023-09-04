## ---- test-fitAutoReg-var-p2-exo-fit-lasso
lapply(
  X = 1,
  FUN = function(i,
                 tol,
                 text) {
    message(text)
    set.seed(42)
    constant <- c(1, 1, 1)
    coef <- matrix(
      data = c(
        0.4, 0.0, 0.0, 0.1, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0, 0.2, 0.0,
        0.0, 0.0, 0.6, 0.0, 0.0, 0.3
      ),
      nrow = 3,
      byrow = TRUE
    )
    exo_coef <- matrix(
      data = c(
        0.5, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 0.5
      ),
      nrow = 3
    )
    Y <- dat_p2_exo_yx$Y
    X <- dat_p2_exo_yx$X
    YStd <- StdMat(Y)
    XStd <- StdMat(X[, -1])
    lambdas <- 10^seq(-5, 5, length.out = 100)
    aic <- FitVARLassoSearch(
      YStd = YStd,
      XStd = XStd,
      lambdas = lambdas,
      crit = "ebic",
      max_iter = 10000,
      tol = 1e-5
    )
    bic <- FitVARLassoSearch(
      YStd = YStd,
      XStd = XStd,
      lambdas = lambdas,
      crit = "ebic",
      max_iter = 10000,
      tol = 1e-5
    )
    ebic <- FitVARLassoSearch(
      YStd = YStd,
      XStd = XStd,
      lambdas = lambdas,
      crit = "ebic",
      max_iter = 10000,
      tol = 1e-5
    )
    aic <- OrigScale(
      coef_std = aic,
      Y = Y,
      X = X[, -1]
    )
    bic <- OrigScale(
      coef_std = bic,
      Y = Y,
      X = X[, -1]
    )
    ebic <- OrigScale(
      coef_std = ebic,
      Y = Y,
      X = X[, -1]
    )
    testthat::test_that(
      paste(text, "aic"),
      {
        testthat::expect_true(
          all(
            abs(cbind(coef, exo_coef) - aic) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "bic"),
      {
        testthat::expect_true(
          all(
            abs(cbind(coef, exo_coef) - bic) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "ebic"),
      {
        testthat::expect_true(
          all(
            abs(cbind(coef, exo_coef) - ebic) <= tol
          )
        )
      }
    )
  },
  tol = 0.19, # allow for some bias
  text = "test-fitAutoReg-var-p2-exo-fit-lasso"
)
