#!/usr/bin/env Rscript

library(tidyverse)
library(tidymodels)

args   = commandArgs(trailingOnly=TRUE)
input  = args[1]

dataset = read_tsv(input)

saveRDS(dataset, file = "input_dataset.rds")
