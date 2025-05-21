---
title: "Drug Distribution"
date: 2025-05-21
weight: 4
---

# 4. Infer Drug Distributions
Let's now explore a path illuminated by our newfound understanding of vascular network morphology, enhanced by the power of Gaussian Process (GP) Modelling, to uncover the biological relevance of vessel arrangement. As you know, any drug reaches it's target through vessels, but how exactly do vessel morphology and distribution impact the diffusion of compounds from the vessels into surrounding tissues? To answer, we must correlate the vascular network extracted from histological images with MALDI imaging data through GP modelling. Your aim is to model the spatial distribution of a drug within a tissue sample based on MALDI imaging, where pixel intensity is directly proportional to drug concentration. However, before applying GP modelling to real images, we must validate the model’s accuracy through simulations.

> **Theoretical Background**: A **Gaussian Process** (GP) is a collection of random variables, where any finite subset follows a multivariate normal distribution. This statistical method offers a probabilistic framework for modelling spatial dependencies. It is widely used in spatial statistics and machine learning to infer continuous functions from discrete data points. In this context, GP modelling allows us to estimate the underlying drug distribution function based on observed MALDI intensities.

   - ## 4.1 Generating synthetic vessel images
      To begin, execute `Final_optimized_4_Windows.R` (script downloadable from the user slrenne's erivessel repository). Note that this code only works on windows operating systems.

      As usual, set up your environment. Notably, set a fixed seed for random number generation and import the `Statistical Rethinking` package (detailed intructions for installation may be found on [Richard McElreath's rethinking repository](https://github.com/rmcelreath/rethinking)). We also create a parallel cluster to speed things up.
      ```r
      #r
      set.seed(20241028)
      library(rethinking)
      library(parallel)
      library(plot.matrix)
      library(viridis)
      library(MASS)
      
      num_cores <- detectCores() - 1
      cl <- makeCluster(num_cores)
      ```
      We start by simulating binary vessel masks as square matrices, where pixels with value `1` represent vessel regions and `0` represent any non-vascular tissue.
      ```r
      #r
      test_function <- function(n, circles) {
        mat <- matrix(0, nrow = n, ncol = n)
        for(i in 1:n) {
          for(j in 1:n) {
            for(circle in circles) {
              center <- circle$center
              radius <- circle$radius
              if (sqrt((i - center[1])^2 + (j - center[2])^2) <= radius) {
                mat[i, j] <- 1
              }
            }
          }
        }
        return(mat)
      }
      ```
   
     Here, you may manually set the number of images and matrix size to obtain your desired trade-off between accuraccuracy and computational cost. In this case we are generating `10` masks each having a `20x20` grid with circular vessel regions. The vessel positions and radii are randomly sampled to introduce variability.
      ```r
      #r
      N_img <- 10   # Number of synthetic images
      n <- 20       # Matrix dimensions
   
      circles_list <- vector("list", N_img)
      for(i in 1:N_img) {
        circles_list[[i]] <- list(
          list(center = c(sample(5:15, 1), sample(5:15, 1)), radius = runif(1, 1, 3)),
          list(center = c(sample(5:15, 1), sample(5:15, 1)), radius = runif(1, 1, 3))
        )
      }
      ```
      
      We generate vessel masks in parallel using the defined `test_function()`.
      ```r
      #r
      clusterExport(cl, c("test_function", "n", "circles_list"))

      mats <- parLapply(cl, 1:N_img, function(i) test_function(n, circles_list[[i]]))
      ```
      
   - ## 4.2 Creating Simulated MALDI Images
      Next, we generate simulated MALDI intensity maps based on any predefined function you like, such as an exponential decay model:

      $$
      I(x) = \eta^2 e^{- \frac{d(x)^2}{2\rho^2}}
      $$

      where we manually define the parameters:
        - $d(x)$: the Euclidean distance from vessels  
        - $\eta^2$: the signal variance  
        - $\rho$: the characteristic length scale controlling spatial correlation

      First, compute Euclidean distance matrices for each image in parallel.
      ```r
      #r
      grids <- parLapply(cl, 1:N_img, function(i) expand.grid(X = 1:n, Y = 1:n))
      Dmats <- parLapply(cl, 1:N_img, function(i) as.matrix(dist(grids[[i]], method = "euclidean")))
      ```
      Then, as previously mentioned, define the the kernel function parameters.
      ```r
      #r
      beta <- 5
      etasq <- 2
      rho <- sqrt(0.5)
      ```
      Compute covariance matrices using the radial basis function (RBF) kernel.
      ```r
      #r
      clusterExport(cl, c("beta", "etasq", "rho", "Dmats"))
      Ks <- parLapply(cl, 1:N_img, function(i) {
        etasq * exp(-0.5 * ((Dmats[[i]] / rho)^2)) + diag(1e-9, n*n)
      })
      ```
      Now, sample synthetic MALDI intensities using a GP prior.
      ```r
      #r
      clusterExport(cl, "Ks")
      
      sim_gp <- parLapply(cl, 1:N_img, function(i) {
        MASS::mvrnorm(1, mu = rep(0, n*n), Sigma = Ks[[i]])
      })
      ```
      Generate observed intensity values by adding noise.
      ```r
      #r
      clusterExport(cl, c("sim_gp"))
      sim_y <- parLapply(cl, 1:N_img, function(i) {
        rnorm(n*n, mean = sim_gp[[i]] + beta * as.vector(t(mats[[i]])), sd = 1)
      })
      ```
   - ## 4.3 Inferring drug distribution with GP regression
      Now that we have simulated data, we fit a Gaussian Process model using **Bayesian inference** with **Stan (rethinking package)**.

      First, prepare data for the Bayesian model.
      ```r
      #r
      dat_list <- list(N = n*n)
      for(i in 1:N_img) {
        dat_list[[ paste0("y", i) ]] <- sim_y[[i]]
        dat_list[[ paste0("x", i) ]] <- as.vector(t(mats[[i]]))
        dat_list[[ paste0("Dmat", i) ]] <- Dmats[[i]]
      }
      ```
      Next, define the GP model.
      ```r
      #r
      model_code <- "alist(\n"
      for(i in 1:N_img) {
        model_code <- paste0(model_code,
                             "  y", i, " ~ multi_normal(mu", i, ", K", i, "),\n",
                             "  mu", i, " <- a + b * x", i, ",\n",
                             "  matrix[N, N]:K", i, " <- etasq * exp(-0.5 * square(Dmat", i, " / rho)) + diag_matrix(rep_vector(0.01, N)),\n")
      }
      model_code <- paste0(model_code,
                           "  a ~ normal(0, 1),\n",
                           "  b ~ normal(0, 0.5),\n",
                           "  etasq ~ exponential(2),\n",
                           "  rho ~ exponential(0.5)\n",
                           ")")
      
      model_list <- eval(parse(text = model_code))
      ```
      
      Fit the GP model using Hamiltonian Monte Carlo (HMC) via the `ulam` function from the rethinking package. This approach enables Bayesian inference on vessel spatial organization and drug distribution. Feel free to play around with the following parameters according to your computational resources. 
      - `chains` sets how many independent MCMC chains are run to ensure proper convergence.
      - `cores` sets how many CPU cores are used to parallelize computation and speed up sampling.
      - `iter` sets how manu iterations are run per chain, including warm-up.
      - `warmup` sets how many iterations are used for warm-up (not included in posterior estimates).
        
      ```r
      #r
      GP_N <- ulam(model_list, data = dat_list, chains = 4, cores = num_cores, iter = 600, warmup = 150)
      ```
      Finally, print the model summary.
      ```r
      #r
      print(precis(GP_N))
      
      post <- extract.samples(GP_N)
      ```
   
   - ## 4.4 Validating the Model
      We visualize the inferred vs. true covariance functions by plotting the priors, the actual kernel and the estimated kernels (your posterior samples). Also, remember to stop the cluster to free resources.
      ```r
      #r
      set.seed(08062002)
      
      p.etasq <- rexp(n, rate = 0.5)
      p.rhosq <- rexp(n, rate = 0.5)
      
      plot(NULL, xlim = c(0, max(Dmats[[1]])/3), ylim = c(0, 10),
           xlab = "pixel distance", ylab = "covariance",
           main = "Prior, Actual, and Estimated Kernel")
      
      # Priors
      for(i in 1:20)
        curve(p.etasq[i] * exp(-0.5 * (x/p.rhosq[i])^2),
              add = TRUE, lwd = 6, col = col.alpha(2, 0.5))
      
      # Actual kernel
      curve(etasq * exp(-0.5 * (x/rho)^2), add = TRUE, lwd = 4)
      
      # Estimated kernels
      for(i in 1:20) {
        curve(post$etasq[i] * exp(-0.5 * (x/post$rho[i])^2),
              add = TRUE, col = col.alpha(4, 0.3), lwd = 6)
      }
      stopCluster(cl)
      ```
      If the inferred decay function closely matches the predefined function, it means the GP model accurately recovers known distribution parameters. So, the model's acccuracy is validated and ready to be easily applied to real world histological and MALDI images!