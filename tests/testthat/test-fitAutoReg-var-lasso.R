## ---- test-fitAutoReg-var-lasso
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
    Y <- dat_demo_yx$Y
    X <- dat_demo_yx$X
    Ystd <- StdMat(Y)
    Xstd <- StdMat(X[, -1])
    lambdas <- 10^seq(-5, 5, length.out = 100)
    search <- SearchVARLasso(
      Ystd = Ystd, Xstd = Xstd, lambdas = lambdas,
      max_iter = 10000, tol = 1e-5
    )
    lasso <- OrigScale(
      SelectVARLasso(search, crit = "ebic"),
      Y = Y,
      X = X[, -1]
    )
    phi <- c(
      lasso[1, 1],
      lasso[2, 2],
      lasso[3, 3],
      lasso[1, 4],
      lasso[2, 5],
      lasso[3, 6]
    )
    lasso[1, 1] <- 0
    lasso[2, 2] <- 0
    lasso[3, 3] <- 0
    lasso[1, 4] <- 0
    lasso[2, 5] <- 0
    lasso[3, 6] <- 0
    testthat::test_that(
      paste(text, "SearchVARLasso", "sparsity", "ebic"),
      {
        testthat::expect_true(
          sum(
            lasso
          ) == 0
        )
      }
    )
    testthat::test_that(
      paste(text, "SearchVARLasso", "auto", "ebic"),
      {
        testthat::expect_true(
          all(
            abs(round(phi, digits = 2) - c(0.4, 0.5, 0.6, 0.1, 0.2, 0.3)) <= tol
          )
        )
      }
    )
    SelectVARLasso(search, crit = "aic")
    SelectVARLasso(search, crit = "bic")
  },
  tol = 0.08, # allow for some bias
  text = "test-fitAutoReg-var-lasso"
)
