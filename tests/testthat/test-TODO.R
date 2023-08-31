## ---- test-TODO
lapply(
  X = 1,
  FUN = function(i,
                 text) {
    message(text)
    set.seed(42)
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
    BootCI(pb)
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
    BootCI(pb)

    # Residual bootstrap
    system.time(
      rb <- RBootVAROLS(
        data = dat_p2,
        p = 2,
        B = 10
      )
    )
    rb$est
    BootCI(rb)
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
    BootCI(rb)

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
    BootSE(pb)
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
    BootSE(pb)

    # Residual bootstrap
    system.time(
      rb <- RBootVAROLS(
        data = dat_p2,
        p = 2,
        B = 10
      )
    )
    rb$est
    BootSE(rb)
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
    BootSE(rb)
  },
  text = "test-TODO"
)
