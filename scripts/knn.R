#!/usr/bin/env Rscript

library(tidyverse)
library(tidymodels)

args   = commandArgs(trailingOnly=TRUE)
input  = args[1]
cores  = args[2]
output = args[3]

writeLines('###READING INPUT DATA###')
bp_data = readRDS(input)

writeLines('###SPLITTING DATA###')
bp_data_split = initial_split(bp_data %>% 
                                mutate(exercise_level = factor(exercise_level, levels = c("high", "moderate", "low"))) %>%
                                mutate(family_history = factor(family_history, levels = c("no", "yes"))) %>%
                                mutate(rs2228570 = factor(rs2228570, levels = c("A/A", "A/G", "G/G"))) %>% 
                                mutate(rs1143627 = factor(rs1143627, levels = c("C/C", "C/T", "T/T"))) %>% 
                                mutate(rs1045642 = factor(rs1045642, levels = c("A/A", "A/G", "G/G"))) %>% 
                                mutate(rs7412 = factor(rs7412, levels = c("C/C", "C/T", "T/T"))) %>% 
                                mutate(rs1801133 = factor(rs1801133, levels = c("A/A", "G/A", "G/G"))) %>% 
                                mutate(rs1800795 = factor(rs1800795, levels = c("C/C", "C/G", "G/G"))),  
                                              prop = 0.75)
bp_data_training = training(bp_data_split)
bp_data_testing = testing(bp_data_split)
bp_data_validation = vfold_cv(bp_data_training)
write_rds(bp_data_training, file = 'training_data_knn.rds')
write_rds(bp_data_testing, file = 'testing_data_knn.rds')
write_rds(bp_data_validation, file = 'validation_data_knn.rds')

writeLines('###STRUCTURE###')
knn_model <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

writeLines('###TUNING GRID###')
knn_tuning_grid <- grid_regular(
  neighbors(),
  weight_func(),
  levels = 3)

writeLines('###RECIPE###')
knn_tuning_recipe <- recipe(
  hypertension_risk_indicator ~ ., 
  data = bp_data_training) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors())

writeLines('###WORKFLOW###')
knn_tuning_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(knn_tuning_recipe)

writeLines('###TUNING GRID###')
knn_tuning_results <-
  knn_tuning_workflow %>% 
  tune_grid(
    resamples = bp_data_validation,
    grid = knn_tuning_grid
  )

writeLines('###COLLECTING METRICS###')
tuning_metrics <- knn_tuning_results %>% 
  collect_metrics()
write_tsv(tuning_metrics, file = 'knn_tuning_metrics.tsv')

writeLines('###SELECTING THE BEST TUNING###')
knn_tuning_best_parameters <- knn_tuning_results %>% 
  select_best('rmse')

writeLines('###FINALIZING THE WORKFLOW###')
final_knn_wf <- knn_tuning_workflow %>% 
  finalize_workflow(knn_tuning_best_parameters)

writeLines('###LAST FIT###')
final_knn_fit <- final_knn_wf %>% 
  last_fit(bp_data_split)

writeLines('###FINAL METRICS###')
final_metrics <- final_knn_fit %>% 
  collect_metrics()
write_tsv(final_metrics, file = 'knn_final_metrics.tsv')

writeLines('###PLOTTING PREDICTIONS###')
png('plot_knn.png')
final_knn_fit %>% 
  collect_predictions() %>% 
  ggplot(aes(x=hypertension_risk_indicator, y=.pred))+
  geom_point(alpha= 0.5, color = 'blue')+
  geom_abline(alpha= 0.9, color = 'red')
dev.off()