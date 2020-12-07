args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  N_PROC <- 2
  SEED <- 12345
  JOBID <- system("date +'%Y%m%d%H%M%S'", intern = TRUE)
  USE_PYTHON <- "/home/gemuend/colin/condaenv/bin/python"
} else {
  N_PROC <- as.integer(args[1])
  SEED <- as.integer(args[2])
  JOBID <- as.character(args[3])
  USE_PYTHON <- as.character(args[4])
}


VADER_PATH <- file.path("VaDER")
DIR_OUT <- file.path("results", "JADNI", "vader", "hyperparameter_optimization", JOBID)
F_OUT <- file.path(DIR_OUT, sprintf("grid_search_seed%i.RData", SEED))
DIR_IN <- file.path("data", "JADNI")
F_IN <- file.path(DIR_IN, "JADNI.RData")
PATTERN <- paste(c("^ADAS11$", "^CDRSB$", "^MMSE$"), collapse = "|")
# PATTERN <- paste(c(sprintf("^ADAS_Q%i$", c(1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)), "^CDR_", "^MMSE_"), collapse = "|")
# PATTERN <- paste(c("^ADAS11$", "^CDRSB$", "^MMSE$", "^FAQ$", "^RAVLT\\."), collapse = "|")
# PATTERN <- paste(c("^ADAS_", "^CDR_", "^MMSE_", "^FAQ_"), collapse = "|")
N_SAMPLE <- 1e4 # Inf; 100 takes 90 minutes using 3x100 cores
# N_SAMPLE <- 16
N_PERM <- 1e3
N_EPOCH_PRE <- 10
N_EPOCH <- 50
PARAM_SEED <- 12345
N_FOLD <- 2
N_REP <- 100 # no. of reps for consensus clustering: ideally, set this to 100 if you have enough cores
PARAM_GRID <- list(
  # k = 2:6,
  # n_layer = 1:2,
  # alpha = 1.0,
  # n_node = c(8, 32, 128),
  # learning_rate = 10^(-3:-2),
  # batch_size = 2^(5:6)
  k = 2:10,
  n_layer = 1:2,
  alpha = 1.0,
  n_node = c(8, 16, 32, 64, 128),
  learning_rate = 10^(-4:-2),
  batch_size = 2^(4:6)
)

library(caret)
library(data.table)
library(abind)
library(parallel)
library(caret)
library(gplots)
library(matrixStats)
library(fossil)
library(reticulate)
# call directly after loading reticulate
if (!is.null(USE_PYTHON)) {
  use_python(USE_PYTHON, required = TRUE)
}
dir.create(DIR_OUT, recursive = TRUE)

load_data <- function(f_in, pattern) {
	s <- load(f_in)
	if ("y" %in% s) { # cluster label is present: artificial data
	  X_train <- X
	  PTID <- 1:dim(X)[1]
	} else {
	  # Take time points as follows to minimize missingness but maximize time range:
	  # Keep the overall missingness at max ~60% (see paper on VaDER)
	  X_train <- Xnorm[
	    ,
	    c(1, 2, 3, 4, 5), # select time points
      grep(pattern, dimnames(Xnorm)[[3]]),
      drop = FALSE
    ]
	  cat(sprintf("Missingness: %.3g%%\n", sum(is.na(X_train)) / prod(dim(X_train))))
	  PTID <- dimnames(Xnorm)[[1]]
	}
	W_train <- !is.na(X_train)
	mode(W_train) <- "integer"
	X_train[is.na(X_train)] <- 0 # arbitrary value
	list(X = X_train, W = W_train, ptid = PTID)
}

consensus_clustering <- function(YHAT, k) {
  M <- matrix(0, nrow = length(YHAT[[1]]), ncol = length(YHAT[[1]]))
  for (yhat in YHAT) {
    for (ii in split(1:length(yhat), yhat)) {
      ii <- as.matrix(expand.grid(ii, ii))
      M[ii] <- M[ii] + 1
    }
  }
  cutree(hclust(as.dist(1 - M), method = "complete"), k = k)
}

cross_validate <- function(params, data, weights, groups, n_fold, n_perm, seed = NULL) {
  # cat("begin cross_validate\n", file = "vader_hyperparameter_optimization.out", append = TRUE)
  cat("begin cross_validate\n")
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  adj_rand_index <- mclust::adjustedRandIndex
  
  # unadjusted rand index
  rand_index <- function(
    p, # clusters predicted on the test data using the model that was trained on the training data
    q # clusters inferred directly from the test data
  ) {
    n <- length(p)
    
    # y: cluster assignment vector
    # returns: length(y) * length(y) binary matrix indicating 
    # which points fall into the same cluster. 
    f <- function(y) {
      m <- matrix(rep(y, length(y)), ncol = length(y))
      m == t(m)
    }
    mp <- f(p)
    mq <- f(q)
    a <- (sum(mp & mq) - n) / 2
    b <- sum(!(mp | mq)) / 2
    (a + b) / choose(n, 2)
  }
  prediction_strength <- function(
    p, # clusters predicted on the test data using the model that was trained on the training data
    q # clusters inferred directly from the test data
  ) {
    n <- length(p)
    
    # y: cluster assignment vector
    # returns: length(y) * length(y) binary matrix indicating 
    # which points fall into the same cluster. 
    f <- function(y) {
      m <- matrix(rep(y, length(y)), ncol = length(y))
      m == t(m)
    }
    mp <- f(p)
    mq <- f(q)
    # mpq <- !mq | mp
    mpq <- mq & mp
    min(tapply(1:n, q, function(ii) {
      n_ii <- length(ii)
      (sum(mpq[,ii]) - n_ii) / n_ii / (n - 1)
    }))
  }
  
  cat("Importing VADER.\n")
  VADER <- reticulate::import_from_path("vader", path = VADER_PATH)$VADER
  
  # train the model
  folds <- caret::createFolds(1:nrow(data), n_fold)
  perf <- do.call("rbind", lapply(1:length(folds), function(i) {
    fold <- folds[[i]]
    save_dir <- file.path(
      "temp", 
      paste(sample(letters, 64, replace = TRUE), collapse = ""),
      gsub("-", "minus", paste(unlist(sapply(params, as.character)), collapse = "_"))
    )
    dir.create(save_dir, recursive = TRUE)
    # save path for VADER
    save_path <- file.path(save_dir, "vader.ckpt")
    
    n_hidden <- lapply(params[grep("n_hidden", names(params))], as.integer)
    names(n_hidden) <- NULL
    
    cat("Calling VaDER init.")
    vader_func <- function(X_train, W_train, seed) {
      VADER(
        X_train = X_train,
        save_path = save_path,
        n_hidden = n_hidden,
        k = as.integer(params$k),
        groups = as.integer(groups),
        learning_rate = params$learning_rate,
        batch_size = as.integer(params$batch_size), 
        alpha = params$alpha,
        output_activation = NULL,
        seed = if (is.null(seed)) NULL else as.integer(seed),
        recurrent = TRUE,
        W_train = W_train,
        n_thread = 16L
      )
    }
    
    Y_PRED <- list()
    loss <- rep(0, 4)
    names(loss) <- c(
      "train_reconstruction_loss", 
      "train_latent_loss", 
      "test_reconstruction_loss", 
      "test_latent_loss"
    )
    for (j in 1:N_REP) {
      cat(sprintf("Fold %d Iteration %d starting.\n", i, j))
      vader <- vader_func(
        data[-fold,,, drop = FALSE], 
        weights[-fold,,, drop = FALSE],
        seed = seed + j
      )
      
      cat(sprintf("Fold %d Iteration %d: calling pre_fit.\n", i, j))
      vader$pre_fit(n_epoch = as.integer(N_EPOCH_PRE), verbose = TRUE)

      cat(sprintf("Fold %d Iteration %d: calling fit.\n", j))
      vader$fit(n_epoch = as.integer(N_EPOCH), verbose = TRUE)

      # how many mixture components are effectively used?
      effective_k <- length(table(vader$cluster(data[-fold,,, drop = FALSE])))
      cat(sprintf("Fold %d Iteration %d: calling cluster.\n", i, j))
      Y_PRED[[j]] <- vader$cluster(
        data[fold,,, drop = FALSE],
        weights[fold,,, drop = FALSE]
      )
      cat(sprintf("Fold %d Iteration %d: calling get_loss.\n", i, j))
      test_loss <- vader$get_loss(
        data[fold,,, drop = FALSE], 
        weights[fold,,, drop = FALSE]
      )
      loss <- loss + c(
        vader$reconstruction_loss, 
        vader$latent_loss, 
        test_loss$reconstruction_loss, 
        test_loss$latent_loss
      )
    }
    loss <- loss / N_REP
    cat("Calling consensus_clustering.\n")
    y_pred <- consensus_clustering(Y_PRED, params$k)
    
    vader <- vader_func(
      data[fold,,, drop = FALSE], 
      weights[fold,,, drop = FALSE],
      seed = seed
    )
    vader$pre_fit(n_epoch = as.integer(N_EPOCH_PRE), verbose = TRUE)
    vader$fit(n_epoch = as.integer(N_EPOCH), verbose = TRUE)
    y_true <- vader$cluster(
      data[fold,,, drop = FALSE], 
      weights[fold,,, drop = FALSE]
    )
    
    arindex <- adj_rand_index(y_pred, y_true)
    rindex <- rand_index(y_pred, y_true)
    pstrength <- prediction_strength(y_pred, y_true)
    null <- t(replicate(n_perm, {
      sample_y_pred <- sample(y_pred)
      c(
        rindex = rand_index(sample_y_pred, y_true),
        arindex = adj_rand_index(sample_y_pred, y_true),
        pstrength = prediction_strength(sample_y_pred, y_true)
      )
    }))
    res <- c(
      loss, 
      effective_k = effective_k, 
      rand_index = rindex, 
      rand_index_null = mean(null[,"rindex"]),
      adj_rand_index = arindex, 
      adj_rand_index_null = mean(null[,"arindex"]),
      prediction_strength = pstrength, 
      prediction_strength_null = mean(null[,"pstrength"])
    )
    # delete model-related files
    unlink(save_dir, recursive=TRUE)
    
    res
  }))
  colMeans(perf)
}

explore_grid <- function(
  data,
  weights = NULL,
  groups,
  param_grid,
  n_sample, # how many random samples to take from the grid?
  n_fold,
  n_proc,
  n_perm,
  seed = NULL,
  param_seed = NULL
) {
  
  nms <- names(param_grid)[names(param_grid) != "n_node"]
  paramspace <- data.table(expand.grid(param_grid[nms]))
  layer_configs <- lapply(param_grid$n_layer, function(n_layer) {
    p <- data.table(do.call(expand.grid, lapply(1:n_layer, function(i) {
      param_grid$n_node
    })))
    names(p) <- sprintf("n_hidden%i", 1:n_layer)
    p
  })
  names(layer_configs) <- param_grid$n_layer

  paramspace <- unlist(lapply(1:nrow(paramspace), function(i) {
    p <- as.matrix(cbind(
      paramspace[i,],
      layer_configs[[as.character(paramspace$n_layer[i])]]
    ))
    nms <- colnames(p)
    p <- split(p, row(p))
    p <- lapply(p, function(pi) {
      names(pi) <- nms
      pi
    })
  }), recursive = FALSE)
  keep <- unlist(lapply(paramspace, function(p) {
    b <- TRUE
    if ("n_hidden2" %in% names(p)) {
      b <- b && p["n_hidden2"] < p["n_hidden1"]
    }
    b
  }))
  paramspace <- paramspace[keep]
  if (n_sample < length(paramspace)) {
    if (!is.null(param_seed)) {
      set.seed(param_seed)
    }
    paramspace <- paramspace[sample(length(paramspace), n_sample)]
  }
  for (i in 1:length(paramspace)) {
    paramspace[[i]] <- as.list(paramspace[[i]])
  }
  paramspace <- paramspace[length(paramspace):1]

  minproc <- min(nrow(paramspace), n_proc)
  cat(sprintf("min from nrow(paramspace) and n_proc is %d (it will start that many R procs).\n", minproc))
  flush.console()

  #cl <- makePSOCKcluster(min(nrow(paramspace), n_proc), outfile="", useXDR=FALSE)
  cl <- makeForkCluster(min(nrow(paramspace), n_proc), outfile="", useXDR=FALSE)
  Sys.sleep(3)
  flush.console()
  is_first_iteration <- TRUE
  clusterExport(cl, envir = environment(), varlist = c(
    "data", "paramspace", "cross_validate", "n_fold", "groups", "consensus_clustering",
    "is_first_iteration", "n_perm", "seed", "weights", "n_proc", "N_REP",
    "VADER_PATH", "USE_PYTHON", "N_EPOCH", "N_EPOCH_PRE"
  ))
  clusterEvalQ(cl, {
    library(reticulate)
    if (!is.null(USE_PYTHON)) {
      use_python(USE_PYTHON, required = TRUE)
    }
  })
  cv_func <- function(i) {
# disable sleep
#    if (is_first_iteration) {
#      #sleep_time = 10 * (i %% n_proc)
#      sleep_time = sample.int(10, 1)
#      cat(sprintf("cv_func iteration %d sleeping for %d seconds.\n", i, sleep_time))
#      flush.console()
#      Sys.sleep(sleep_time) # to avoid high CPU loads simultaneously from all processes
#      cat(sprintf("Waking up in cv_func iteration %d.\n", i))
#      flush.console()
#      is_first_iteration <- FALSE
#    }
    params <- paramspace[[i]]
    # # make sure each worker is seeded differently, but deterministically depends on the master thread
    # if (!is.null(seed)) { 
    #   seed <- sample(.Machine$integer.max, i)[i]
    # }
    # cat("begin call cross_validate\n", file = "vader_hyperparameter_optimization.out", append = TRUE)
    cat("begin call cross_validate.\n")
    res <- cross_validate(params, data, weights, groups, n_fold, n_perm, seed = seed)
    cat(
      sprintf(
        "%i of %i: \teffective_k=%.2g \t[%s]\n",
        i, length(paramspace), res["effective_k"],
        paste(sprintf("%s=%.4f", names(params), as.numeric(params)), collapse = "; ")
      )
    )
    res
  }
  environment(cv_func) <- .GlobalEnv 
  len_x <- length(paramspace)
  cat(sprintf("Calling rbind clusterApplyLB, length of x is %d.\n", len_x))
  perf <- do.call("rbind", clusterApplyLB(cl, 1:length(paramspace), cv_func))
#  perf <- do.call("rbind", clusterApplyLB(cl, 1:10, cv_func))
  cat("Finished.\n")
  stopCluster(cl)
  paramspace <- do.call("rbind", c(lapply(paramspace, function(params) {
    data.table(t(unlist(params)))
  }), fill = TRUE))
  dt <- data.table(cbind(paramspace, perf))
  setorderv(dt, cols = colnames(dt)[1:tail(grep("n_hidden", colnames(dt)), 1)])
}

L <- load_data(f_in = F_IN, pattern = PATTERN)
GROUPS <- as.integer(factor(gsub("_.*", "", dimnames(L$X)[[3]]))) - 1
perf <- explore_grid(
  data = L$X,
  weights = L$W,
  groups = GROUPS,
  param_grid = PARAM_GRID,
  n_sample = N_SAMPLE,
  n_fold = N_FOLD,
  n_proc = N_PROC,
  n_perm = N_PERM,
  seed = SEED,
  param_seed = PARAM_SEED
)
save(perf, file = F_OUT)
