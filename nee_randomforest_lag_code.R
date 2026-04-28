
install.packages("ranger")
install.packages("tidymodels")


# ------ Load packages -----
library(tidyverse)
library(tidymodels)
library(lubridate)
#--------------------------#


# --Model description--- #
# Add a brief description of your modeling approach


# -- Uncertainty representation -- #
# Describe what sources of uncertainty are included in your forecast and how you estimate each source.



# Change this for your model ID
# Include the word "example" in my_model_id for a test submission
# Don't include the word "example" in my_model_id for a forecast that you have registered (see neon4cast.org for the registration form)
my_model_id <- 'nee_randfor_lag'


# Read data #
# read in the targets data
url <- "https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/project_id=neon4cast/duration=P1D/terrestrial_daily-targets.csv.gz"
terrestrial_targets <- readr::read_csv(url, show_col_types = FALSE)

#read in the sites data
site_data <- readr::read_csv("https://raw.githubusercontent.com/eco4cast/neon4cast-targets/main/NEON_Field_Site_Metadata_20220412.csv") |> 
  dplyr::filter(terrestrial == 1)

focal_sites <- site_data$field_site_id


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

# Future weather forecast --------
# New forecast only available at 5am UTC the next day
forecast_date <- Sys.Date() 
noaa_date <- forecast_date - days(1)

weather_future_s3 <- neon4cast::noaa_stage2(start_date = as.character(noaa_date-1)) #changed to noaadate-1 because there was error pulling in data

weather_future <- weather_future_s3 |> 
  dplyr::filter(datetime >= forecast_date,
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

#plot nee data for fun
#ggplot(targets_lm, aes(x = datetime, y = nee)) + geom_line()



forecast_horizon <- 30
forecasted_dates <- seq(from = ymd(forecast_date), to = ymd(forecast_date) + forecast_horizon, by = "day")
n_members <- 200



forecast_df <- NULL

# ----- Fit model & generate forecast----

for (s in 1:length(focal_sites)){
  
  # pull in targets 
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
    
    weather_ensemble_names <- unique(weather_future_daily$parameter)
    
  #-----Constructing Random Forest
  
  #pre-process data
  set.seed(100)
  split <- initial_split(targets_lm, prop = 0.80, strata = site_id)
  
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
    fit(data = train_nee)
  
  #----process uncertainty: residual SD from training fit ------
  train_preds <- predict(nee_full_fit, new_data = train_nee)$.pred
  resid_sd <- sd(train_nee$nee - train_preds, na.rm = TRUE)
  
  ggplot(pred_test, aes(x = nee, y = .pred)) + geom_point()
  
  #add initial conditions uncertainty
  dates_2025 <- seq(as.Date("2025-01-01"), as.Date("2025-1-31"), by = "day")
  targets_unc <- targets_lm |> filter(as.Date(datetime) %in% dates_2025)
  initc_sd <- sd(targets_unc$nee)
  initc_sd_df <- rnorm(n_members, mean = 0, sd = initc_sd)



  #----------Make a forecast--------------#
  
  prev_nee <- tail(targets_lm$nee, 1) + initc_sd_df
  # Loop through all forecast dates
  for (t in 1:length(forecasted_dates)) {
  
    #loop over each ensemble member
    met_ens_id <- 0
    for(ens in 1:n_members){
      if(met_ens_id <= 30){
        met_ens_id <- met_ens_id + 1
        ens_nm <- paste0(ens, "-", met_ens_id)
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
      
      forecast_df <- bind_rows(forecast_df,
                               tibble(
                                 datetime = forecasted_dates[t],
                                 reference_datetime = forecast_date,
                                 site_id = focal_sites[s],
                                 ensemble = ens,
                                 parameter = met_ens,
                                 prediction = forecast_pred,
                                 variable = "nee"
                               ))
      
      
  }
}
}

#plot forecasts
ggplot(data = forecast_df, mapping = aes(x = datetime, y = prediction, group = ensemble)) +
  geom_line()+
  facet_wrap(~ site_id)




#---- Convert to EFI standard ----

# Make forecast fit the EFI standards
forecast_df_EFI <- forecast_df %>%
  filter(datetime > forecast_date) %>%
  mutate(model_id = my_model_id,
         reference_datetime = forecast_date,
         family = 'ensemble',
         duration = 'P1D',
         parameter = as.character(parameter),
         project_id = 'neon4cast') %>%
  select(datetime, reference_datetime, duration, site_id, family, parameter, variable, prediction, model_id, project_id)
#---------------------------#



# ----- Submit forecast -----
# Write the forecast to file
theme <- 'terrestrial'
date <- forecast_df_EFI$reference_datetime[1]
forecast_name <- paste0(forecast_df_EFI$model_id[1], ".csv")
forecast_file <- paste(theme, date, forecast_name, sep = '-')

write_csv(forecast_df_EFI, forecast_file)

neon4cast::forecast_output_validator(forecast_file)


neon4cast::submit(forecast_file =  forecast_file, ask = FALSE) # if ask = T (default), it will produce a pop-up box asking if you want to submit

#--------------------------#
