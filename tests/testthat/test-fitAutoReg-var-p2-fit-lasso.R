## ---- test-fitAutoReg-var-p2-fit-lasso
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
    null_mat <- matrix(
      data = 0,
      nrow = nrow(coef),
      ncol = ncol(coef)
    )
    coef_vec <- c(
      coef[1, 1],
      coef[2, 2],
      coef[3, 3],
      coef[1, 4],
      coef[2, 5],
      coef[3, 6]
    )
    Y <- dat_p2_yx$Y
    X <- dat_p2_yx$X
    YStd <- StdMat(Y)
    XStd <- StdMat(X[, -1])
    lambdas <- 10^seq(-5, 5, length.out = 100)
    aic <- FitVARLassoSearch(
      YStd = YStd,
      XStd = XStd,
      lambdas = lambdas,
      crit = "aic",
      max_iter = 10000,
      tol = 1e-5
    )
    bic <- FitVARLassoSearch(
      YStd = YStd,
      XStd = XStd,
      lambdas = lambdas,
      crit = "bic",
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
    aic_sparse <- aic
    aic_sparse[1, 1] <- 0
    aic_sparse[2, 2] <- 0
    aic_sparse[3, 3] <- 0
    aic_sparse[1, 4] <- 0
    aic_sparse[2, 5] <- 0
    aic_sparse[3, 6] <- 0
    bic_sparse <- bic
    bic_sparse[1, 1] <- 0
    bic_sparse[2, 2] <- 0
    bic_sparse[3, 3] <- 0
    bic_sparse[1, 4] <- 0
    bic_sparse[2, 5] <- 0
    bic_sparse[3, 6] <- 0
    ebic_sparse <- ebic
    ebic_sparse[1, 1] <- 0
    ebic_sparse[2, 2] <- 0
    ebic_sparse[3, 3] <- 0
    ebic_sparse[1, 4] <- 0
    ebic_sparse[2, 5] <- 0
    ebic_sparse[3, 6] <- 0
    aic_coef <- c(
      aic[1, 1],
      aic[2, 2],
      aic[3, 3],
      aic[1, 4],
      aic[2, 5],
      aic[3, 6]
    )
    bic_coef <- c(
      bic[1, 1],
      bic[2, 2],
      bic[3, 3],
      bic[1, 4],
      bic[2, 5],
      bic[3, 6]
    )
    ebic_coef <- c(
      ebic[1, 1],
      ebic[2, 2],
      ebic[3, 3],
      ebic[1, 4],
      ebic[2, 5],
      ebic[3, 6]
    )
    testthat::test_that(
      paste(text, "FitVARLassoSearch", "sparsity", "aic"),
      {
        testthat::expect_true(
          all(
            abs(null_mat - aic_sparse) <= 0.05
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "FitVARLassoSearch", "sparsity", "bic"),
      {
        testthat::expect_true(
          all(
            abs(null_mat - bic_sparse) <= 0.05
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "FitVARLassoSearch", "sparsity", "ebic"),
      {
        testthat::expect_true(
          all(
            abs(null_mat - ebic_sparse) <= 0.05
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "FitVARLassoSearch", "coef", "aic"),
      {
        testthat::expect_true(
          all(
            abs(coef_vec - aic_coef) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "FitVARLassoSearch", "coef", "bic"),
      {
        testthat::expect_true(
          all(
            abs(coef_vec - bic_coef) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "FitVARLassoSearch", "coef", "ebic"),
      {
        testthat::expect_true(
          all(
            abs(coef_vec - ebic_coef) <= tol
          )
        )
      }
    )
  },
  tol = 0.1, # allow for some bias
  text = "test-fitAutoReg-var-p2-fit-lasso"
)
