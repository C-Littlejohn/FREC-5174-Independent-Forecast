

# ------ Load packages -----
library(tidyverse)
library(tidymodels)
library(lubridate)
library(scoringRules)
library(purrr)
library(duckdbfs)
#--------------------------#

my_model_id <- 'nee_randfor_lag'

#-----Read data ---------
# read in the targets data
url <- "https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/project_id=neon4cast/duration=P1D/terrestrial_daily-targets.csv.gz"
terrestrial_targets <- readr::read_csv(url, show_col_types = FALSE)

#read in the sites data
site_data <- readr::read_csv("https://raw.githubusercontent.com/eco4cast/neon4cast-targets/main/NEON_Field_Site_Metadata_20220412.csv") |> 
  dplyr::filter(terrestrial == 1)

focal_sites <- site_data$field_site_id[1:2]

# Filter the targets
targets <- terrestrial_targets |> 
  filter(site_id %in% focal_sites, variable == "nee")

#--------------------------#


# ------ Weather data ------
met_variables <- c("air_temperature", "precipitation_flux", "surface_downwelling_shortwave_flux_in_air")

# Past stacked weather -----
weather_past_daily <- data.frame()
for(i in 1:length(focal_sites)){
  focal_site = focal_sites[i]

  weather_past_s3 <- neon4cast::noaa_stage3()

  weather_past <- weather_past_s3  |> 
    dplyr::filter(site_id == focal_site,
                  datetime >= ymd('2017-01-01'),
                  variable %in% met_variables) |> 
    dplyr::collect()

  # aggregate the past to mean values
  weather_past_sub <- weather_past |> 
    mutate(datetime = as_date(datetime)) |> 
    group_by(datetime, site_id, variable) |> 
    summarize(prediction = mean(prediction, na.rm = TRUE), .groups = "drop") |> 
    # convert air temperature to Celsius if it is included in the weather data
    mutate(prediction = ifelse(variable == "air_temperature", prediction - 273.15, prediction)) |> 
    pivot_wider(names_from = variable, values_from = prediction)
  
  weather_past_daily <- rbind(weather_past_daily, weather_past_sub)
}

#--------------------#



forecast_horizon <- 30

n_members <- 200

forecast_df <- NULL

#set reference datetime
ref_datetime <- seq(as.Date("2025-01-01"), as.Date("2025-12-31"), by = "30 day")


#loop through each site

for (s in 1:length(focal_sites)){
  
  # Generate a dataframe to fit the model to 
  targets_lm <- targets |> 
    pivot_wider(names_from = 'variable', values_from = 'observation') |> 
    left_join(weather_past_daily, 
              by = c("datetime","site_id")) |> 
    filter(site_id %in% focal_sites[s]) |> 
    na.omit() |> 
    #add nee lag to targets
    arrange(site_id, datetime) |> 
    group_by(site_id) |> 
    mutate(nee_lag = lag(nee, 1)) |> 
    ungroup()
  
  curr_site <- focal_sites[s]
  site_target <- targets_lm |> 
    filter(site_id == curr_site)
  
  for(rd in 1:length(ref_datetime)){
    
    print(ref_datetime[rd])
    training_data_trimmed <- site_target |> 
      filter(datetime < ref_datetime[rd])
    forecasted_dates <- seq(from = ref_datetime[rd], to = ref_datetime[rd] + days(forecast_horizon), by = "day")
    
    
  
    #weather future
    forecast_date <- as.Date(ref_datetime[rd])
    noaa_date <- forecast_date - days(1)
    
    datetimes <- seq(ref_datetime[rd], ref_datetime[rd] + days(30))
    
    weather_future_s3 <- neon4cast::noaa_stage2(start_date = as.character(noaa_date))
    
    weather_future <- weather_future_s3 |> 
      dplyr::filter(datetime %in% datetimes,
                    site_id %in% focal_sites,
                    variable %in% met_variables) |> 
      collect()
    
    weather_future_daily <- weather_future |> 
      mutate(datetime = as_date(datetime)) |> 
      # mean daily forecasts at each site per ensemble
      group_by(datetime, site_id, parameter, variable) |> 
      summarize(prediction = mean(prediction, na.rm = TRUE), .groups = "drop") |> 
      # convert air temperature to Celsius if it is included in the weather data
      mutate(prediction = ifelse(variable == "air_temperature", prediction - 273.15, prediction)) |> 
      pivot_wider(names_from = variable, values_from = prediction) |> 
      select(any_of(c('datetime', 'site_id', met_variables, 'parameter')))
    
    #--------------------------#
  
    weather_ensemble_names <- unique(weather_future_daily$parameter)
  
  #-----Constructing Random Forest
    
    #pre-process data
    set.seed(100)
    split <- initial_split(training_data_trimmed, prop = 0.80)
    
    train_nee <- training(split)
    test_nee <- testing(split)
    
    #split training data into folds
    
    folds <- vfold_cv(train_nee, v = 5)
    
    
    #feature engineering using a recipe
    nee_recipe <- train_nee |> 
      recipe(nee ~ . ) |> 
      step_rm(datetime, site_id, duration) |> 
      step_naomit(air_temperature, precipitation_flux, surface_downwelling_shortwave_flux_in_air, nee_lag)
    
    #hyperparameters tuning
    rand_for_mod <- 
      rand_forest(min_n = tune(), trees = tune(), mtry = tune()) |> 
      set_engine("ranger", num.threads = parallel::detectCores()) |> 
      set_mode("regression")
    
    #workflow
    nee_wflow <- 
      workflow() |> 
      add_model(rand_for_mod) |> 
      add_recipe(nee_recipe)
    
    # --------train model---------
    nee_resample_fit <- 
      nee_wflow |>  
      tune_grid(resamples = folds,
                grid = 25,
                control = control_grid(save_pred = TRUE),
                metrics = metric_set(rmse))
    
    nee_resample_fit |>  
      collect_metrics() |> 
      arrange(mean)
    
    best_hyperparameters <- nee_resample_fit |> 
      select_best(metric = "rmse")
    
    best_hyperparameters
    
    #update workflow with best hyper-parameters
    final_workflow <- 
      nee_wflow |> 
      finalize_workflow(best_hyperparameters)
    
    #fit to all training data
    nee_fit <- final_workflow |> 
      fit(data = train_nee)
    
    #predict testing data
    predictions <- predict(nee_fit, 
                           new_data = test_nee)
    pred_test <- bind_cols(test_nee, predictions)
    
    #evaluate model
    multi_metric <- metric_set(rmse, rsq)
    
    metric_table <- pred_test |> 
      multi_metric(truth = nee, estimate = .pred)
    
    metric_table
    
    #train model on full dataset
    nee_full_fit <- final_workflow |> 
      fit(data = training_data_trimmed)
    
    ggplot(pred_test, aes(x = nee, y = .pred)) + geom_point()
    
    #process uncertainty: residual SD from training fit ------
    train_preds <- predict(nee_full_fit, new_data = train_nee)$.pred
    resid_sd <- sd(train_nee$nee - train_preds, na.rm = TRUE)
    
    #add initial conditions uncertainty
    dates_2025 <- seq(as.Date("2025-01-01"), as.Date("2025-1-31"), by = "day")
    targets_unc <- targets_lm |> filter(as.Date(datetime) %in% dates_2025)
    initc_sd <- sd(targets_unc$nee)
    initc_sd_df <- rnorm(n_members, mean = 0, sd = initc_sd)
    
    
    #----------Make a forecast--------------#
    
    prev_nee <- tail(targets_lm$nee, 1) + initc_sd_df
    # Loop through all forecast dates
    for (t in 1:length(datetimes)) {
     
      current_preds <- numeric(n_members)
      
       #loop over each ensemble member
      met_ens_id <- 0
      for(ens in 1:n_members){
        print(paste(rd, "-", t, "-", ens))
        if(met_ens_id <= 30){
          met_ens_id <- met_ens_id + 1
        }else{
          met_ens_id <- 1
        }
        
        met_ens <- weather_ensemble_names[met_ens_id]
        
        #add initial condition uncertainty to nee_lag 
        pred_df <- weather_future_daily %>%
          filter(datetime == as.Date(forecasted_dates[t]),
                 site_id == focal_sites[s],
                 parameter == met_ens
          ) |> 
          mutate(nee_lag = prev_nee[ens],
                 project_id = "neon4cast",
                 duration = "P1D")
        
        #generate prediction
        forecast_pred <- predict(nee_full_fit, new_data = pred_df)$.pred + rnorm(1, 0, resid_sd)
        
        current_preds[ens] <- forecast_pred
        
         forecast_df <- bind_rows(forecast_df,
                                 tibble(
                                   datetime = forecasted_dates[t],
                                   reference_datetime = forecast_date,
                                   site_id = focal_sites[s],
                                   ensemble = ens,
                                   parameter = met_ens,
                                   prediction = forecast_pred,
                                   variable = "nee"))

      } #close ensemble loop
      
      prev_nee <- current_preds
      
    } #close datetimes loop
  } #close refdatetime loop
} #close site loop


#plot forecasts
ggplot(data = forecast_df, mapping = aes(x = datetime, y = prediction, group = ensemble)) +
  geom_line() +
  facet_wrap(~ site_id)


#---------Crps of reforecast------------

# add observations
forecast_eval <- forecast_df  |> 
  left_join(targets  |>  select(datetime, site_id, observation, variable),
            by = c("datetime", "site_id", "variable"))  |> 
  rename(truth = observation)  |> 
  drop_na(truth)

#compute horizon
forecast_eval <- forecast_eval  |> 
  mutate(horizon = as.numeric(as.Date(datetime) - as.Date(reference_datetime)))


#calculate CRPS for my model
crps_my_model <- forecast_eval  |> 
  group_by(site_id, datetime, reference_datetime, horizon)  |> 
  summarize(
    crps = scoringRules::crps_sample(
      y = first(truth),
      dat = prediction),
    .groups = "drop") |> 
  mutate(model_id = "nee_randfor_lag")


#bring in baseline model info
baseline_models <- arrow::open_dataset("s3://anonymous@bio230014-bucket01/challenges/scores/bundled-parquet/project_id=neon4cast/duration=P1D/variable=nee?endpoint_override=sdsc.osn.xsede.org") |> 
  filter(reference_datetime > lubridate::as_datetime("2025-01-01"),
         reference_datetime < lubridate::as_datetime("2025-12-31"),
         model_id %in% c("climatology", "persistenceRW")) |> 
  collect() 

#combine CRPS from my model and baselines
baseline_crps <- baseline_models |> 
  group_by(model_id, datetime, reference_datetime) |> 
  slice(1) |> 
  ungroup() |>
  mutate(horizon = as.numeric(datetime - reference_datetime)) |> 
  select(model_id, horizon, datetime, reference_datetime, crps)

my_crps_summary <- crps_my_model %>%
  group_by(model_id, horizon) %>%
  summarize(mean_crps = mean(crps), .groups = "drop")

baseline_summary <- baseline_crps %>%
  group_by(model_id, horizon) %>%
  summarize(mean_crps = mean(crps, na.rm = TRUE), .groups = "drop")

combined_crps <- bind_rows(baseline_summary, my_crps_summary)



#plot all models mean crps by horizon
p_horizon <- ggplot(data = combined_crps, mapping = aes(x = horizon, y = mean_crps, color = model_id)) + 
  geom_line() +
  labs(x = "Horizon (days)", y = "mean CRPS", title = "CRPS by horizon") +
  theme_bw()

#plot crps by time
my_crps_time <- crps_my_model |> 
  group_by(model_id, datetime) |> 
  summarize(mean_crps = mean(crps), .groups = "drop")
baseline_time <- baseline_models |> 
  group_by(model_id, datetime) |> 
  summarize(mean_crps = mean(crps, na.rm = TRUE), .groups = "drop")
crps_time_all <- bind_rows(my_crps_time, baseline_time)

p_time <- ggplot(crps_time_all, aes(x = datetime, y = mean_crps, color = model_id)) +
  geom_line() +
  theme_bw() +
  labs(x = "Date", y = "CRPS", title = "CRPS over time")

#plot crps by nlcd class aka ecosystem type
crps_with_class <- crps_my_model |> 
  left_join(
    site_data |> 
      select(field_site_id, field_dominant_nlcd_classes),
    by = c("site_id" = "field_site_id")
  )

crps_h7 <- crps_with_class |> 
  filter(horizon == 7)

crps_nlcd <- crps_h7 |> 
  group_by(field_dominant_nlcd_classes) |> 
  summarize(mean_crps = mean(crps, na.rm = TRUE), .groups = "drop")

p_nlcd <- ggplot(crps_nlcd, aes(x = field_dominant_nlcd_classes, y = mean_crps)) +
  geom_col() + 
  theme_bw() + 
  labs(x = "NLCD Class", y = "CRPS Horizon 7", title = "Forecast CRPS by land cover type")


#---------------------------#

#save key datasets to csv

write_csv(crps_my_model, "crps_my_model_full.csv")
write_csv(baseline_models, "crps_baseline_models_full.csv")
write_csv(combined_crps, "crps_combined_summary.csv")

write_csv(forecast_df, "forecast_ensembles.csv")

#save plots as image files

ggsave("crps_by_horizon.png", p_horizon, width = 8, height = 5, dpi = 300)
ggsave("crps_over_time.png", p_time, width = 10, height = 5, dpi = 300)
ggsave("crps_by_nlcd.png", p_nlcd, width = 8, height = 5, dpi = 300)




