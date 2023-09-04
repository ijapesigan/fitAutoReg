## ---- test-TODO
lapply(
  X = 1,
  FUN = function(i,
                 text) {
    message(text)
    set.seed(42)
    # Trigger the warning in FitVARLasso
    YStd <- scale(iris[, c("Sepal.Length", "Sepal.Width")])
    XStd <- scale(iris[, c("Petal.Length", "Petal.Width")])
    lambda <- 73.90722
    FitVARLasso(
      YStd = YStd,
      XStd = XStd,
      lambda = lambda,
      max_iter = 2,
      tol = 1e-5
    )

    # Parametric bootstrap
    system.time(
      pb <- PBootVAROLS(
        data = dat_p2,
        p = 2,
        B = 10,
        burn_in = 20
      )
    )
    pb$est
    print(BootCI(pb))
    print(BootSE(pb))

    system.time(
      pb <- PBootVARLasso(
        data = dat_p2,
        p = 2,
        B = 10,
        burn_in = 20,
        n_lambdas = 100,
        crit = "ebic",
        max_iter = 1000,
        tol = 1e-5
      )
    )
    pb$est
    print(BootCI(pb))
    print(BootSE(pb))

    # Residual bootstrap
    system.time(
      rb <- RBootVAROLS(
        data = dat_p2,
        p = 2,
        B = 10
      )
    )
    rb$est
    print(BootCI(rb))
    print(BootSE(rb))

    system.time(
      rb <- RBootVARLasso(
        data = dat_p2,
        p = 2,
        B = 10,
        n_lambdas = 100,
        crit = "ebic",
        max_iter = 1000,
        tol = 1e-5
      )
    )
    rb$est
    print(BootCI(rb))
    print(BootSE(rb))

    # Parametric bootstrap (Exogenous)
    data <- dat_p2_exo$data
    exo_mat <- dat_p2_exo$exo_mat
    system.time(
      pb <- PBootVARExoOLS(
        data = data,
        exo_mat = exo_mat,
        p = 2,
        B = 5,
        burn_in = 0
      )
    )
    pb$est
    print(BootCI(pb))
    print(BootSE(pb))

    system.time(
      pb <- PBootVARExoLasso(
        data = data,
        exo_mat = exo_mat,
        p = 2,
        B = 5,
        burn_in = 0,
        n_lambdas = 10,
        crit = "ebic",
        max_iter = 1000,
        tol = 1e-5
      )
    )
    pb$est
    print(BootCI(pb))
    print(BootSE(pb))

    # Residual bootstrap (Exogenous)
    system.time(
      rb <- RBootVARExoOLS(
        data = data,
        exo_mat = exo_mat,
        p = 2,
        B = 5
      )
    )
    rb$est
    print(BootCI(rb))
    print(BootSE(rb))

    system.time(
      rb <- RBootVARExoLasso(
        data = data,
        exo_mat = exo_mat,
        p = 2,
        B = 5,
        n_lambdas = 10,
        crit = "ebic",
        max_iter = 1000,
        tol = 1e-5
      )
    )
    rb$est
    print(BootCI(rb))
    print(BootSE(rb))
  },
  text = "test-TODO"
)
