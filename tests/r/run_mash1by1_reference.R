#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: run_mash1by1_reference.R <repo_root> <bhat_csv> <shat_csv> <alpha> <out_dir>")
}

repo_root <- args[[1]]
bhat_csv <- args[[2]]
shat_csv <- args[[3]]
alpha <- as.numeric(args[[4]])
out_dir <- args[[5]]

suppressPackageStartupMessages({
  library(ashr)
  library(mvtnorm)
  library(plyr)
  library(abind)
  library(assertthat)
  library(softImpute)
  library(rmeta)
})

r_dir <- file.path(repo_root, "mashr", "R")
files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
for (f in files) {
  source(f)
}

Bhat <- as.matrix(read.csv(bhat_csv, header = FALSE))
Shat <- as.matrix(read.csv(shat_csv, header = FALSE))

data <- mash_set_data(Bhat = Bhat, Shat = Shat, alpha = alpha)
m <- mash_1by1(data, alpha = alpha)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.table(m$result$PosteriorMean, file.path(out_dir, "posterior_mean.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$result$PosteriorSD, file.path(out_dir, "posterior_sd.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$result$lfsr, file.path(out_dir, "lfsr.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(m$loglik, file.path(out_dir, "loglik.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
