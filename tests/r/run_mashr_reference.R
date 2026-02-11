#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(4, 5, 6))) {
  stop("Usage: run_mashr_reference.R <repo_root> <bhat_csv> <shat_csv> <out_dir> [pi_csv] [optmethod]")
}

repo_root <- args[[1]]
bhat_csv <- args[[2]]
shat_csv <- args[[3]]
out_dir <- args[[4]]
pi_csv <- if (length(args) >= 5) args[[5]] else NA_character_
if (!is.na(pi_csv) && toupper(pi_csv) == "NA") {
  pi_csv <- NA_character_
}
optmethod <- if (length(args) >= 6) args[[6]] else "mixEM"

r_dir <- file.path(repo_root, "mashr", "R")
if (dir.exists(r_dir)) {
  suppressPackageStartupMessages({
    library(ashr)
    library(mvtnorm)
    library(plyr)
    library(abind)
    library(assertthat)
    library(softImpute)
    library(rmeta)
  })

  files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (f in files) {
    source(f)
  }
} else {
  suppressPackageStartupMessages(library(mashr))
}

Bhat <- as.matrix(read.csv(bhat_csv, header = FALSE))
Shat <- as.matrix(read.csv(shat_csv, header = FALSE))

# Match Python test configuration exactly for comparison.
data <- mash_set_data(Bhat = Bhat, Shat = Shat)
U <- cov_canonical(data)
if (!is.na(pi_csv)) {
  pi_vec <- as.numeric(read.csv(pi_csv, header = FALSE)[[1]])
  g <- list(pi = pi_vec, Ulist = U, grid = c(0.5, 1.0), usepointmass = TRUE)
  m <- mash(
    data,
    g = g,
    fixg = TRUE,
    verbose = FALSE,
    algorithm.version = "R",
    outputlevel = 2,
    output_lfdr = TRUE
  )
} else {
  m <- mash(
    data,
    Ulist = U,
    grid = c(0.5, 1.0),
    prior = "uniform",
    optmethod = optmethod,
    verbose = FALSE,
    algorithm.version = "R",
    outputlevel = 2,
    output_lfdr = TRUE
  )
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.table(m$fitted_g$pi, file.path(out_dir, "pi.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$result$PosteriorMean, file.path(out_dir, "posterior_mean.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$result$PosteriorSD, file.path(out_dir, "posterior_sd.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$result$lfsr, file.path(out_dir, "lfsr.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$loglik, file.path(out_dir, "loglik.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
