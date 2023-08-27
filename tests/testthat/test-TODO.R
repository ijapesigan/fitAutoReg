## ---- test-TODO
lapply(
  X = 1,
  FUN = function(i,
                 text) {
    message(text)
    set.seed(42)
    system.time(
      pb <- PBootVAROLS(data = dat_p2, p = 2, B = 10, burn_in = 20)
    )
    pb$est
    PBootCI(pb)
    PBootSE(pb)
    system.time(
      pb <- PBootVARLasso(
        data = dat_p2, p = 2, B = 10, burn_in = 20,
        n_lambdas = 100, crit = "ebic", max_iter = 1000, tol = 1e-5
      )
    )
    pb$est
    PBootCI(pb)
    PBootSE(pb)
  },
  text = "test-TODO"
)
