#!/usr/bin/env Rscript

library(tidyverse)
library(tidymodels)
library(vip)

args   = commandArgs(trailingOnly=TRUE)
input  = args[1]
cores  = args[2]
output = args[3]

writeLines('###READING INPUT DATA###')
blood_pressure_data = readRDS(input)

writeLines('###SPLITTING DATA###')
blood_pressure_data_split = initial_split(blood_pressure_data %>% 
                                            mutate(exercise_level = factor(exercise_level, levels = c("high", "moderate", "low"))) %>%
                                            mutate(family_history = factor(family_history, levels = c("no", "yes"))) %>%
                                            mutate(rs2228570 = factor(rs2228570, levels = c("A/A", "A/G", "G/G"))) %>% 
                                            mutate(rs1143627 = factor(rs1143627, levels = c("C/C", "C/T", "T/T"))) %>% 
                                            mutate(rs1045642 = factor(rs1045642, levels = c("A/A", "A/G", "G/G"))) %>% 
                                            mutate(rs7412 = factor(rs7412, levels = c("C/C", "C/T", "T/T"))) %>% 
                                            mutate(rs1801133 = factor(rs1801133, levels = c("A/A", "G/A", "G/G"))) %>% 
                                            mutate(rs1800795 = factor(rs1800795, levels = c("C/C", "C/G", "G/G"))),  
                                          prop = 0.75)
blood_pressure_data_training = training(blood_pressure_data_split)
blood_pressure_data_testing = testing(blood_pressure_data_split)
blood_pressure_data_validation = vfold_cv(blood_pressure_data_training)
write_rds(blood_pressure_data_training, file = 'training_data_rf.rds')
write_rds(blood_pressure_data_testing, file = 'testing_data_rf.rds')
write_rds(blood_pressure_data_validation, file = 'validation_data_rf.rds')

writeLines('###STRUCTURE###')
rf_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>% 
  set_engine('ranger') %>% 
  set_mode('regression')

writeLines('###TUNING GRID###')
rf_tuning_grid <- grid_regular(
  mtry(range = c(5L,8L)),
  trees(),
  min_n(),
  levels = 3
)

writeLines('###RECIPE###')
rf_tuning_recipe <- recipe(
  hypertension_risk_indicator ~ ., 
  data = blood_pressure_data_training) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors())

writeLines('###TUNING WORKFLOW###')
rf_tuning_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(rf_tuning_recipe)

writeLines('###TUNING RESULTS###')
rf_tuning_results <- rf_tuning_workflow %>% 
  tune_grid(
    resamples = blood_pressure_data_validation,
    grid = rf_tuning_grid
  )

writeLines('###COLLECTING METRICS###')
tuning_metrics <- rf_tuning_results %>% 
  collect_metrics()
write_tsv(tuning_metrics, file = 'rf_tuning_metrics.tsv')

writeLines('###PLOTTING THE TUNING###')
png('tuning_plot.png')
tuning_metrics %>% 
  ggplot(aes(x=trees, y=mean, color = factor(min_n)))+
  geom_line(linewidth = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(.metric ~ mtry, ncol = 3) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)
dev.off()

writeLines('###SELECTING THE BEST TUNING###')
rf_tuning_best_params <- rf_tuning_results %>% 
  select_best('rmse')

writeLines('###FINALIZING THE WORKFLOW###')
final_rf_wf <- rf_tuning_workflow %>% 
  finalize_workflow(rf_tuning_best_params)

writeLines('###FINAL FIT###')
final_rf_fit <- final_rf_wf %>% 
  last_fit(blood_pressure_data_split)

writeLines('###FINAL METRICS###')
final_metrics <- final_rf_fit %>% 
  collect_metrics()
write_tsv(final_metrics, file = 'rf_final_metrics.tsv')

writeLines('###PLOTTING PREDICTIONS###')
png('plot_random_forest.png')
final_rf_fit %>% 
  collect_predictions() %>% 
  ggplot(aes(x=hypertension_risk_indicator, y=.pred))+
  geom_point(alpha= 0.5, color = 'blue')+
  geom_abline(alpha= 0.9, color = 'red')
dev.off()

writeLines('###IMPORTANCE PLOT###')
rf_tuning_best_model <- finalize_model(
  rf_model,
  rf_tuning_best_params)

png('importance_plot_random_forest.png')
rf_tuning_best_model %>% 
  set_engine('ranger', importance = 'permutation') %>% 
  fit(hypertension_risk_indicator ~ .,
      data = blood_pressure_data_testing) %>% 
  vip(geom = 'point')
dev.off()
