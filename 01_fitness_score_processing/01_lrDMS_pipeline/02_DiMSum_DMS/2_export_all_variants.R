#!/usr/bin/env Rscript
#
# Writes each gate run's all_variants table out as CSV.
#
# DiMSum's fitness stage does not write this table to disk. It leaves
# DiMSum_Project_fitness_replicates.RData, which holds all_variants alongside the wildtype,
# synonymous, singles and doubles tables; only the last four are written out, as the
# fitness_*.txt files. all_variants is the one the fitness score processing in 02a_ and 02b_
# reads, so it has to be exported here, and this step closes the gap between a DiMSum run and
# those folders.
#
# all_variants is the widest of the five: every variant that survived DiMSum's filters, not
# just the singles and doubles, with per-replicate fitness and sigma columns as well as the
# combined ones. For the lenient gate that is 1,113 rows against the 476 of the four tables
# DiMSum does write.
#
# R is needed to read an R object; DiMSum already requires it, so anything that can produce
# these runs can run this. No packages beyond base R.
#
# Run after 1_run_dimsum.py. Reads, per gate folder:
#     <gate>/DiMSum_Project_fitness_replicates.RData
# Writes:
#     <gate>/all_variants.csv

GATES <- c("lenient_gate", "stringent_gate")
RDATA_NAME <- "DiMSum_Project_fitness_replicates.RData"
OBJECT_NAME <- "all_variants"

script_path <- sub("^--file=", "",
                   grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
this_dir <- if (length(script_path)) normalizePath(dirname(script_path)) else getwd()

missing <- character(0)

for (gate in GATES) {
  source_path <- file.path(this_dir, gate, RDATA_NAME)
  target_path <- file.path(this_dir, gate, "all_variants.csv")

  if (!file.exists(source_path)) {
    cat(sprintf("[%s] %s not found\n", gate, RDATA_NAME))
    missing <- c(missing, gate)
    next
  }

  loaded <- new.env()
  objects <- load(source_path, envir = loaded)
  if (!(OBJECT_NAME %in% objects)) {
    cat(sprintf("[%s] %s holds no '%s' (it holds: %s)\n",
                gate, RDATA_NAME, OBJECT_NAME, paste(objects, collapse = ", ")))
    missing <- c(missing, gate)
    next
  }

  table <- as.data.frame(get(OBJECT_NAME, envir = loaded))
  write.csv(table, target_path)
  cat(sprintf("[%s] %d variants, %d columns -> %s\n",
              gate, nrow(table), ncol(table), basename(target_path)))
}

if (length(missing)) {
  cat(sprintf("\n%d of %d gates could not be exported\n", length(missing), length(GATES)))
  quit(status = 1)
}
cat(sprintf("\n%d gates exported\n", length(GATES)))
