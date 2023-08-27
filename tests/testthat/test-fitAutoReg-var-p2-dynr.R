## ---- test-fitAutoReg-var-p2-dynr
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
    dynr_model <- ModelVARP2Dynr(
      data = dat_p2
    )
    dynr_cook <- FitVARDynr(model = dynr_model)
    coefs <- coef(
      object = dynr_model,
      dynr_cook = dynr_cook,
      hessian_flag = FALSE
    )
    constant_est <- round(coefs$constant, digits = 0)
    coef_est <- round(coefs$coef, digits = 1)
    testthat::test_that(
      paste(text, "constant"),
      {
        testthat::expect_true(
          all(
            abs(
              constant - constant_est
            ) <= tol
          )
        )
      }
    )
    testthat::test_that(
      paste(text, "coef"),
      {
        testthat::expect_true(
          all(
            abs(
              coef - coef_est
            ) <= tol
          )
        )
      }
    )
  },
  tol = 0.05,
  text = "test-fitAutoReg-var-p2-dynr"
)
