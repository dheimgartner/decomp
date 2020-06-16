# Script description ----
# Containing all the functions of the package. The package's decomposition routines are 
# mainly built by importing and reorganizing functionality from other packages. The package
# also delivers a shapley value regression function for simply regression models and a
# sigmoidal roi-curve fit routine



# --------------------------------------------------------
# shap_decomposition ----
# Description:      Decomposes any model locally with help of the game theoretic shapley values.
#                   Handles any type of model with a suitable prediction function.
#                   The supplied prediction function takes two argmuents: the model object and
#                   a new data frame. Leveraging the python shapper library.
#              
# Parameters:
# - model         = Model object
# - feat_dec      = Character vector of features to be decomposed
# - data          = Data frame available to modeler
# - target        = Character of target variable in data
# - predictors    = Character vector of all the features
# - pfun          = Prediction function(model, newdata) and returns vector of predictions
# 
# Output:
# - contributions = Data frame with dimensions nrow = nrow(data) and ncol = length(feat_dec)
# --------------------------------------------------------

shap_decomposition <- function(model, feat_dec, data, target, predictors, pfun) {
  
  # Initial requirements --
  if(!require(shapper)) stop("This function requires the shapper package!")
  if(!require(DALEX)) stop("This function requires the DALEX package!")
  if(!reticulate::py_module_available("shap")) stop("This function requires the py_module shap!")
  
  
  # Build explainer --
  # Black-box models may have very different structures. This function creates a unified 
  # representation of a model, which can be further processed by various explainers.
  explainer <-
    DALEX::explain(model = model, data = data[, predictors], y = data[, target],
                   predict_function = pfun, verbose = F)
  
  
  # Compute Shapley values --
  # Main function of shapper package; for each observation we perform shap value
  # decomposition and store the results in a list
  shap_list <- list()
  for (i in 1:nrow(data)) {
    shap_list[[i]] <- shap(explainer, new_observation = data[i, predictors])}
  
  
  # Reorganize --
  # The shap values are stored in the _attribution_ column (long format...)
  contributions <- data[, feat_dec]
  for (i in 1:length(feat_dec)) {
    contributions[, feat_dec[i]] <- map(.x = shap_list,
                                          .f = ~.x$`_attribution_`[i]) %>% unlist()
  }
  
  return(contributions)
}



# --------------------------------------------------------
# ale_decomposition ----
# Description:      Decomposes any model locally with help of accumulated local effects.
#                   Handles any type of model with a suitable prediction function.
#                   The supplied prediction function takes two argmuents: the model object and
#                   a new data frame.
#              
# Parameters:
# - model         = Model object
# - feat_dec      = Character vector of features to be decomposed
# - data          = Data frame available to modeler
# - target        = Character of target variable in data
# - pfun          = Prediction function(model, newdata) and returns vector of predictions
# - grid          = grid.size in FeatureEffects function
# 
# Output:
# - contributions = Data frame with dimensions nrow = nrow(data) and ncol = length(feat_dec)
# --------------------------------------------------------

ale_decomposition <- function(model, feat_dec, data, target, pfun, grid = 30) {
  
  # Initial requirements --
  if(!require("iml")) stop("This function requires the package iml")
  if(!is.character(feat_dec) && !is.character(target)) stop("The arguments feat_dec and target are required to be of class character")
  if(!is.data.frame(data)) stop("The data argument has to be of class data.frame")
  
  
  
  # Build predictor container --
  # A Predictor object holds any machine learning model (mlr, caret, randomForest, ...) 
  # and the data to be used for analyzing the model. The interpretation methods in the 
  # iml package need the machine learning model to be wrapped in a Predictor object.
  data <- as.data.frame(data)
  predictor <- Predictor$new(data = data, model = model, y = data[,target], predict.function = pfun)
  
  
  
  # Compute ALE --
  # grid.size governs the trade-off between smoothness and number of datapoints within
  # each interval. If grid size is smaller than nrows(data) then this results in NA values 
  # after merging the contributions and the data frame...
  ale <- FeatureEffects$new(predictor, method = "ale", grid.size = grid)
  
  # Extract ale estimates
  ale <- ale$results %>% .[feat_dec]
  ale <- map2(.x = ale, .y = feat_dec, .f = ~rename(.x, !!.y := .borders))
  ale <- map(.x = ale, .f = ~left_join(data, .x)[, ".value"])
  
  # Reorganize --
  contributions <- reduce(ale, cbind)
  colnames(contributions) <- feat_dec
  contributions <- as_tibble(contributions)
  
  return(contributions)
}



# --------------------------------------------------------
# add_noise ----
# Description:      Adds noise according to noise ratio. The function targets the specified
#                   noise ratio which is defined as var(col) / var(noise).
#              
# Parameters:
# - x             = Vector for which you want to add noise
# - noise_ratio   = Specify noise ratio: target = var(col) / var(noise)
# 
# Output:
# - x + noise     = Returns the input vector with added noise.
# --------------------------------------------------------

add_noise <- function(x, noise_ratio) {
  noise <- rnorm(x)
  k <- sqrt(var(x) / (noise_ratio * var(noise)))
  x_noise <- x + k * noise
  return(x_noise)
}



# --------------------------------------------------------
# fit_roi ----
# Description:      Fits a curve with desirable properties to the scatter in the spend -
#                   contribution plane and thus yields the ROI-curve. The curves are forced
#                   to pass through the origin.
#              
# Parameters:
# - data          = Data frame with scatter (x-value spend) and contributions
# - col_x         = Specify x (spend) col
# - col_y         = Specify y (contribution) col
# - domain        = Vector specifying the domain over which the ROI-curve will be defined
# - fit           = Character specifying the curve to be fit ("logis" or "hill")
# - noise_ratio   = If the nls fit does not converge we can add noise according to this ratio
# 
# Output:
# - list          = returns a list with the (x, y) values of the fitted curve over the defined
#                   domain being one element and the parameter values of the fitted curve
#                   being the second element.
# --------------------------------------------------------

fit_roi <- function(data, col_x, col_y, domain, fit, noise_ratio) {
  
  # Initial requirements --
  if(!exists("add_noise")) stop("This function requires the function add_noise in the global environment.")
  if(!is.character(col_x) && !is.character(col_y)) stop("col_x and col_y have to be of type character")
  if(!require("minpack.lm")) stop("This function requires the package minpack.lm!")
  if(!require("drc")) stop("This function requires the package drc")
  
  data <- as.data.frame(data)
  
  
  # Logis --
  # Create formula
  if (fit == "logis") {
    formula_1 <- paste0("SSlogis(", col_x, ", Asym, xmid, scal)")
    formula_1 <- paste(col_y, formula_1, sep = "~")
    
    
    # Fit curve
    logis_fit <- tryCatch(nlsLM(formula = formula_1, data = data), error = function(e) NA)
    
    # If the fit did not converge, we add noise to the data points and fit again. The
    # magnitude of added noise is governed by th noise ratio. Here, we leverage the
    # add_noise function defined above.
    if (is_na(logis_fit)) {
      
      # Add noise
      data[, col_y] <- add_noise(data[, col_y], noise_ratio)
      
      # Fit again
      logis_fit <- tryCatch(nlsLM(formula = formula_1, data = data), error = function(e) NA)
    }
    
    # Predict over domain --
    # Recall, that the domain is a function argument. For each point in the domain we
    # predict its y value, which is our ROI-curve...
    if (!is_na(logis_fit)) {
      
      # Predict
      domain <- data.frame(domain)
      names(domain) <- col_x
      out <- predict(logis_fit, newdata = domain)
      
      # Force curve to go through the origin
      shift <- out[1]
      out <- out - shift
      
      # Extract parameters of final fit (going through the origin). This is not solved too
      # elegantly: we fit again a curve to the already smooth but shifted function in order
      # to get its parameter values.
      formula_2 <- paste0("SSlogis(", col_x, ", Asym, xmid, scal)")
      formula_2 <- paste("out", formula_2, sep = "~")
      
      newdata <- cbind(domain, out)
      logis_fit <- tryCatch(nlsLM(formula = formula_2, data = newdata), error = function(e) NA)
      
      if (!is_na(logis_fit)) {parameters <- summary(logis_fit)$coefficients}
    }
    
    
    
    # Hill --
    # Very similar to above but now we fit a hill curve to the data.
  } else if (fit == "hill") {
    
    # Fit curve
    formula_1 <- paste(col_y, col_x, sep = "~")
    hill_fit <- tryCatch(drm(formula = formula_1, fct = LL.4(), data = data), error = function(e) NA)
    
    # If the fit did not converge, we add noise to the data points and fit again. The
    # magnitude of added noise is governed by th noise ratio. Here, we leverage the
    # add_noise function defined above.
    if (is_na(hill_fit)) {
      
      # Add noise
      data[, col_y] <- add_noise(data[, col_y], noise_ratio)
      
      # Fit again
      hill_fit <- tryCatch(drm(formula = formula_1, fct = LL.4(), data = data), error = function(e) NA)
    }
    
    # Predict over domain --
    # Recall, that the domain is a function argument. For each point in the domain we
    # predict its y value, which is our ROI-curve...
    if (!is_na(hill_fit)) {
      
      # Predict
      domain <- data.frame(domain)
      names(domain) <- col_x
      out <- predict(hill_fit, newdata = domain)
      
      # Force curve to go through the origin
      shift <- out[1]
      out <- out - shift
      
      # Extract parameters of final fit (going through the origin). This is not solved too
      # elegantly: we fit again a curve to the already smooth but shifted function in order
      # to get its parameter values.
      formula_2 <- paste("out", col_x, sep = "~")
      
      newdata <- cbind(domain, out)
      hill_fit <- tryCatch(drm(formula = formula_2, fct = LL.4(), data = newdata), error = function(e) NA)
      
      if (!is_na(hill_fit)) {parameters <- summary(hill_fit)$coefficients}
    }
  } else {stop("Your fit parameter is wrongly specified.")}
  
  
  # Organize --
  fit <- ifelse(fit == "logis", logis_fit, hill_fit)
  
  # If it converged (after to tries with added noise)
  if (!is_na(fit)) {
    ret <- list(curve = out, params = parameters)
    
    # If it did not converge, we return objects of suitable length containing NA values
  } else {
    ret <- list(curve = rep(NA, length(domain)), params = NA)
  }
  
  return(ret)
}



# --------------------------------------------------------
# min_shap ----
# Description:      Function needed in the shapley_value reg function. In particular in the
#                   optimization routine (vgl. Mishra paper)
#              
# Parameters:
# - correlation   = Pair-wise correlation vector among regressors
# - correlationresponse = Pair-wise correlation vector between regressand and regressors 
# - par           = The choice variable (to be minimized)
# - shapleyscore  = The shapley value vector
#
# Output:
# - function      = Returns an objective function to be minimized
# --------------------------------------------------------

min_shap <- function(correlation, correlationresponse, par, shapleyscore){
  
  # This formula corresponds to the minimization routine in the methodology section SVR 
  # of our thesis. The relevant equation is (21)
  matrix <- (2 * correlationresponse - correlation %*% par)
  f <- sum((par * matrix - shapleyscore)^2)
  
  return(f)
}



# --------------------------------------------------------
# shapley_value_reg ----
# Description:      Applies the game theoretic concept of shapley values to regression
#                   analysis by weighting the regression coefficients accorcing to the
#                   shapley values. In particular, shapley value regression is supposed
#                   to outperform regular OLS in case of high multicollinearity in the 
#                   feature space and complex interaction structures (synergies).
#                   Currently, our function works for log-lin, log-log and regular
#                   regression specifications.
#              
# Parameters:
# - data          = Data frame available to the modeler (both dependent and independent vars)
# - predictors    = Character vector with col names of predictors (features)
# - target        = Character: dependent variable
# - always_in     = Parameter for optimization routine
# - log_dependent = Logical: Should dependent var be log transformed? Set to T
# - log_predictors = Logical: Should predictors be log transformed? Set to T
# - reg           = Logical: Perform regular regression? Set to F
# - reg_formula   = Formula specifying the regression specification if reg was set to T
#
# Output:
# - list          = Returns a list with name coefficients (shapley coefficients),
#                   shapley_value (SV for the predictors), VIF (variable importance factor)
#                   and formula (the regression specification).
# --------------------------------------------------------

shapley_value_reg <- function(data, predictors, target, always_in = NULL, log_dependent = T, 
                              log_predictors = T, reg = F, reg_formula = NULL) {
  
  # Initial requirements --
  if(!exists("min_shap")) stop("This function requires the function minimizeshapley in the global environment.")
  if(!require("relaimpo")) stop("This function requires the package relaimpo!")
  
  # Preparation -- 
  # Keep an unscaled version of the data (later to be used when rescaling the regression
  # coefficients and computing the intercept)
  unscaled_data <- data
  
  # Keep an unscaled verison of the data for reg transformed data set. We need a
  # column for each coefficient.
  if (reg) {
    unscaled_data <- model.frame(reg_formula, data)
    transformed_predictors <- names(unscaled_data)[-1]
  }
  
  # If log predictors is TRUE then add constant of 1 to predictors (to avoid log of zero) 
  # and log transform
  if (log_predictors) {
    data <- data %>% mutate_at(predictors, function(x) x + 1) %>% mutate_at(predictors, log)
  }
  
  # If log target is TRUE then add constant of 1 to target and log transform
  if (log_dependent) {
    data <- data %>% mutate_at(target, function(x) x + 1) %>% mutate_at(target, log)
  }
  
  
  # Creat formula --
  rhs <- str_c(predictors, collapse = " + ")
  lhs <- target
  formula <- str_c(lhs, rhs, sep = " ~ ")
  
  # If reg is TRUE then overwrite formula with reg_formula
  if (reg) {formula <- reg_formula}
  
  
  # Reorganize data --
  # Prepare for regression: select the relevant columns
  data <- data[, c(target, predictors)]
  
  # Add variables according to reg_formula. This is probably not necessary as the lm
  # command transforms the date on its own... But maybe because of scaling?
  if (reg) {data <- model.frame(formula, data)}
  
  # Scale the data
  data <- mutate_all(data, ~as.numeric(scale(.)))
  
  
  # Modeling --
  fit <- lm(formula, data)
  
  # Calculate the feature importance in terms of shapley values. Here, we leverage the
  # relaimp package which stands for relative importance (of the respective feature). It
  # basically computes the shapley values with the R2 function being the value function.
  fit_shap <- calc.relimp(fit, type = "lmg", rank = F, always = always_in)
  
  # These are the shapley values for the predictors (numeric)
  shapley <- fit_shap@lmg
  
  
  # Optimal coefficients --
  # Here we follow closely the procedure described in Mishra (2016) and outlined in the thesis.
  if (reg) {
    
    # These include the higher polynomial column names. For each coefficient ther is a column.
    predictors <- transformed_predictors
    data_scaled_poly <- model.frame(reg_formula, data)
    
    # Correlation matrix among predictors
    correlation <- as.matrix(cor(data_scaled_poly[-1]))
    
    # Correlation vector between target and predictors
    correlation_vector <- cor(data_scaled_poly[-1], data[, target])[, 1]
  
    } else {
    correlation <- as.matrix(cor(data[ ,predictors]))
    correlation_vector <- cor(data[,predictors], data[[target]])[, 1]
  }
  
  # Initial parameter values used in optimization
  initial_parameter <- shapley/correlation_vector
  
  
  # Optimization --
  # Minimize the R2 sum of shapley value difference (see thesis formula). The minimizeshapley
  # function is defined in the mmm_function script (Nepa), but corresponds basically to the
  # formula referred to above.
  optimal <- optim(par = initial_parameter, fn = min_shap, method =  "BFGS",
                   correlationresponse = correlation_vector, correlation = correlation, 
                   shapleyscore = shapley, control = list(maxit = 1000))
  
  # Assign opt parameter values to variable
  optimal_coeff <- optimal$par
  
  
  # Scale back --
  # The computations outlined here follow Mishra (2016):
  # Scale back the optimal parameter values to obtain regression coefficients of the unscaled
  # data. We have to consider the three cases log_dependent, log_dependent and log_predictors,
  # reg separately.
  sd_predictors <- sapply(unscaled_data[, predictors, drop = F], sd)
  
  # log-lin case
  if (log_dependent & !log_predictors) {
    sd_log_target <- sapply(log(unscaled_data[, target, drop = F] + 1), sd)
    regular_beta <- optimal_coeff*(sd_log_target/sd_predictors)
  
    # log-log case
    } else if (log_dependent & log_predictors) {
    sd_log_target <- sapply(log(unscaled_data[, target, drop = F] + 1), sd)
    sd_log_predictors <- sapply(log(unscaled_data[, predictors, drop = F] + 1), sd)
    regular_beta <- optimal_coeff*(sd_log_target/sd_log_predictors)
  
    # reg case
    } else {
    sd_target <- sapply(log(unscaled_data[, target, drop = F]), sd)
    regular_beta <- optimal_coeff*(sapply(unscaled_data[, target, drop = F], sd)/sd_predictors)
  }
  
  
  # Compute the Intercept --
  # Again, we follow Mishra (2016) and distinguish the three cases
  mean_predictors <- sapply(unscaled_data[, predictors, drop = F], mean)
  
  # log-lin case
  if (log_dependent & !log_predictors) {
    mean_log_target <- sapply(log(unscaled_data[, target, drop = F] + 1), mean)
    intercept <- mean_log_target - sum(mean_predictors * regular_beta)
  
    # log-log case
    } else if (log_dependent & log_predictors) {
    mean_log_target <- sapply(log(unscaled_data[, target, drop = F] + 1), mean)
    mean_log_predictors <- sapply(log(unscaled_data[, predictors, drop = F] + 1), mean)
    intercept <- mean_log_target - sum(mean_log_predictors * regular_beta)
  
    # reg case
    } else {
    mean_target <- sapply(unscaled_data[, target, drop = F], mean)
    intercept <- mean_target - sum(mean_predictors * regular_beta)
  }
  
  
  # Reorganize --
  # Combine the intercept and the rescaled beta values
  coefficients <- unlist(list("(Intercept)" = intercept, regular_beta[predictors]))
  names(coefficients) <- c("(Intercept)", predictors)
  
  # Variable inflation factor
  VIF <- variance_inflation_factor(data, formula)
  
  # Return the model
  return(list("coefficients" = coefficients, "shapley_value" = shapley, "VIF" = VIF,
              "formula" = formula))
}
