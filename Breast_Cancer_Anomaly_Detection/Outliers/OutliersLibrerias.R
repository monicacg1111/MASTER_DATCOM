# Master -> Detección de anomalías
# Juan Carlos Cubero. Universidad de Granada

library(tidyverse)  
library(fitdistrplus)  # Ajuste de una distribución -> denscomp 
library(ggpubr)    # ggqqplot -para los gráficos QQ-
library(ggbiplot)  # biplot
library(outliers)  # Grubbs
library(MVN)       # mvn: Test de normalidad multivariante  
library(CerioliOutlierDetection)  #MCD Hardin Rocke
library(mvoutlier) # corr.plot 
library(DDoutlier) # lof
library(cluster)   # PAM
library(heplots)

