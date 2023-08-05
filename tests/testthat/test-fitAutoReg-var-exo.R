## ---- test-fitAutoReg-var-exo
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
    Y <- dat_demo_exo_yx$Y
    X <- dat_demo_exo_yx$X
    coef_est <- FitVAROLS(Y = Y, X = X)
    coef_est[, 1] <- round(coef_est[, 1], digits = 0)
    coef_est[, -1] <- round(coef_est[, -1], digits = 2)
    testthat::test_that(
      paste(text, "constant and coef"),
      {
        testthat::expect_true(
          all(
            abs(
              cbind(
                constant,
                coef,
                exo_coef
              ) - coef_est
            ) <= tol
          )
        )
      }
    )
  },
  tol = 0.10,
  text = "test-fitAutoReg-var-exo"
)
