#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop('Usage: run_cov_ed_reference.R <bhat_csv> <shat_csv> <out_dir>')
}

bhat_csv <- args[[1]]
shat_csv <- args[[2]]
out_dir <- args[[3]]

suppressPackageStartupMessages(library(mashr))

Bhat <- as.matrix(read.csv(bhat_csv, header = FALSE))
Shat <- as.matrix(read.csv(shat_csv, header = FALSE))

data <- mash_set_data(Bhat, Shat)
U_init <- cov_pca(data, npc = 2)
res <- mashr:::bovy_wrapper(data, U_init, maxiter = 80, tol = 1e-5)

# Save outputs
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
write.table(res$pi, file.path(out_dir, 'pi.csv'), sep = ',', row.names = FALSE, col.names = FALSE)

K <- length(res$Ulist)
R <- nrow(res$Ulist[[1]])
arr <- array(0, dim = c(K, R, R))
for (k in 1:K) arr[k,,] <- res$Ulist[[k]]

# one matrix per file for simple readback
for (k in 1:K) {
  write.table(arr[k,,], file.path(out_dir, paste0('U_', k, '.csv')), sep = ',', row.names = FALSE, col.names = FALSE)
}
write.table(res$av_loglik, file.path(out_dir, 'av_loglik.csv'), sep = ',', row.names = FALSE, col.names = FALSE)
