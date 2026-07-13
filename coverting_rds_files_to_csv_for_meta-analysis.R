library(metafor)

### MODELWISE
epi_wMAE_mv_modelwise      <- readRDS("model_epi_wMAE_mv_modelwise.rds")
epi_pearson_mv_modelwise   <- readRDS("model_epi_pearson_mv_modelwise.rds")
brain_wMAE_mv_modelwise    <- readRDS("model_brain_wMAE_mv_modelwise.rds")
brain_pearson_mv_modelwise <- readRDS("model_brain_pearson_mv_modelwise.rds")

write.csv(data.frame(
  model = gsub("^model", "", rownames(epi_wMAE_mv_modelwise$beta)),
  wMAE  = exp(epi_wMAE_mv_modelwise$beta[, 1]),
  ci.lb = exp(epi_wMAE_mv_modelwise$ci.lb),
  ci.ub = exp(epi_wMAE_mv_modelwise$ci.ub)
), file.path("model_epi_wMAE_mv_modelwise.csv"), row.names = FALSE)

write.csv(data.frame(
  model     = gsub("^model", "", rownames(epi_pearson_mv_modelwise$beta)),
  pearson_r = transf.ztor(epi_pearson_mv_modelwise$beta[, 1]),
  ci.lb     = transf.ztor(epi_pearson_mv_modelwise$ci.lb),
  ci.ub     = transf.ztor(epi_pearson_mv_modelwise$ci.ub)
), file.path("model_epi_pearson_mv_modelwise.csv"), row.names = FALSE)

write.csv(data.frame(
  model = gsub("^model", "", rownames(brain_wMAE_mv_modelwise$beta)),
  wMAE  = as.numeric(brain_wMAE_mv_modelwise$beta),
  ci.lb = brain_wMAE_mv_modelwise$ci.lb,
  ci.ub = brain_wMAE_mv_modelwise$ci.ub
), file.path("model_brain_wMAE_mv_modelwise.csv"), row.names = FALSE)

write.csv(data.frame(
  model     = gsub("^model", "", rownames(brain_pearson_mv_modelwise$beta)),
  pearson_r = transf.ztor(brain_pearson_mv_modelwise$beta[, 1]),
  ci.lb     = transf.ztor(brain_pearson_mv_modelwise$ci.lb),
  ci.ub     = transf.ztor(brain_pearson_mv_modelwise$ci.ub)
), file.path("model_brain_pearson_mv_modelwise.csv"), row.names = FALSE)


### GLOBAL
epi_wMAE_mv_global        <- readRDS("model_epi_wMAE_mv_global.rds")
epi_pearson_mv_global     <- readRDS("model_epi_pearson_mv_global.rds")
brain_wMAE_mv_global      <- readRDS("model_brain_wMAE_mv_global.rds")
brain_pearson_mv_global   <- readRDS("model_brain_pearson_mv_global.rds")

write.csv(data.frame(model="Pooled",
                     wMAE=exp(epi_wMAE_mv_global$beta[1,1]),
                     ci.lb=exp(epi_wMAE_mv_global$ci.lb[1]),
                     ci.ub=exp(epi_wMAE_mv_global$ci.ub[1])
), "model_epi_wMAE_mv_global.csv", row.names=FALSE)

write.csv(data.frame(model="Pooled",
                     pearson_r=transf.ztor(epi_pearson_mv_global$beta[1,1]),
                     ci.lb=transf.ztor(epi_pearson_mv_global$ci.lb[1]),
                     ci.ub=transf.ztor(epi_pearson_mv_global$ci.ub[1])
), "model_epi_pearson_mv_global.csv", row.names=FALSE)

write.csv(data.frame(model="Pooled",
                     wMAE=as.numeric(coef(brain_wMAE_mv_global)),
                     ci.lb=as.numeric(brain_wMAE_mv_global$ci.lb),
                     ci.ub=as.numeric(brain_wMAE_mv_global$ci.ub)
), "model_brain_wMAE_mv_global.csv", row.names=FALSE)

write.csv(data.frame(model="Pooled",
                     pearson_r=transf.ztor(brain_pearson_mv_global$beta[1,1]),
                     ci.lb=transf.ztor(brain_pearson_mv_global$ci.lb[1]),
                     ci.ub=transf.ztor(brain_pearson_mv_global$ci.ub[1])
), "model_brain_pearson_mv_global.csv", row.names=FALSE)




#### Also saving relevant file for Fig 4A (reusing Marlene's code from USED_plots_main_040626.R, just saving output as .csv)

##### USED association modelwise - legend inside (4A) ------

results_dir = "/Users/vb506/Documents/dashboard/data"
  
# load results needed
model_assoc_mv_global<- readRDS(
  file.path(
    results_dir,
    "model_assoc_mv_global.rds"
  )
)

model_assoc_mv_modelwise <- readRDS(
  file.path(
    results_dir,
    "model_assoc_mv_modelwise.rds"
  )
)



# 1. Pairwise pooled estimates

pair_est <- tibble(
  term  = rownames(model_assoc_mv_modelwise$b),
  b     = as.numeric(model_assoc_mv_modelwise$b),
  se    = model_assoc_mv_modelwise$se,
  ci.lb = model_assoc_mv_modelwise$ci.lb,
  ci.ub = model_assoc_mv_modelwise$ci.ub,
  pval  = model_assoc_mv_modelwise$pval
) %>%
  mutate(term = str_remove(term, "^model_combi")) %>%
  separate(term, into = c("brain_model", "epi_model"), sep = "\\.")


#mod2_eff2 <- readRDS(file.path(results_dir, "mod2_eff2.rds")) # added this in from line 30
mod2_eff2 <- read.csv(file.path(results_dir, "mod2_eff2.csv")) # local verison is .csv so changed accordingly

pair_raw <- mod2_eff2 %>%
  select(cohort, brain_model, epi_model, RLM_Estimate_scaled)

# 2. Define epi clock groups and desired order

gen1_models <- sort(c(
  "AltumAge", "CorticalClock", "Hannum", "Horvath2013",
  "PCBrainAge", "PedBE", "Wu", "ZhangBLUP", "ZhangEN",
  "cAge", "skinHorvath"
))

gen2plus_models <- sort(c(
  "AdaptAge", "DamAge", "DNAmTL", "DunedinPACE",
  "PCGrimAge", "PhenoAge"
))

# We want TOP -> BOTTOM:
# Gen1 alphabetical, then Gen2plus alphabetical, then Pooled
# For ggplot y-axis, levels must be bottom -> top:
epi_levels <- c(
  "Pooled",
  rev(gen2plus_models),
  rev(gen1_models)
)

# 3. Marginal pooled effects

# pooled across epi clocks for each brain model
mod_brain_margin <- rma.mv(
  yi = RLM_Estimate_scaled,
  V  = RLM_SE_coeftest_HC3_scaled^2,
  mods = ~ brain_model - 1,
  random = list(
    ~ 1 | cohort/timepoint,
    ~ 1 | epi_model
  ),
  data = mod2_eff2,
  method = "REML"
)

brain_margin_est <- tibble(
  brain_model = rownames(mod_brain_margin$b),
  epi_model   = "Pooled",
  b           = as.numeric(mod_brain_margin$b),
  ci.lb       = mod_brain_margin$ci.lb,
  ci.ub       = mod_brain_margin$ci.ub
) %>%
  mutate(
    brain_model = str_remove(brain_model, "^brain_model")
  )

# pooled across brain models for each epi model
mod_epi_margin <- rma.mv(
  yi = RLM_Estimate_scaled,
  V  = RLM_SE_coeftest_HC3_scaled^2,
  mods = ~ epi_model - 1,
  random = list(
    ~ 1 | cohort/timepoint,
    ~ 1 | brain_model
  ),
  data = mod2_eff2,
  method = "REML"
)

epi_margin_est <- tibble(
  epi_model   = rownames(mod_epi_margin$b),
  brain_model = "Pooled",
  b           = as.numeric(mod_epi_margin$b),
  ci.lb       = mod_epi_margin$ci.lb,
  ci.ub       = mod_epi_margin$ci.ub
) %>%
  mutate(
    epi_model = str_remove(epi_model, "^epi_model")
  )

# grand pooled effect
grand_est <- tibble(
  brain_model = "Pooled",
  epi_model   = "Pooled",
  b           = model_assoc_mv_global$b[1,1],
  ci.lb       = model_assoc_mv_global$ci.lb[1],
  ci.ub       = model_assoc_mv_global$ci.ub[1]
)

# 4. Combine all estimates

plot_est <- bind_rows(
  pair_est %>% select(brain_model, epi_model, b, ci.lb, ci.ub),
  brain_margin_est,
  epi_margin_est,
  grand_est
)

# Export for dashboard hybrid approach
write.csv(
  plot_est %>%
    mutate(brain_model = as.character(brain_model),
           epi_model   = as.character(epi_model)),
  file.path(results_dir, "assoc_plot_est.csv"),
  row.names = FALSE
)
