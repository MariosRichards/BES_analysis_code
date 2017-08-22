library(foreign)

dataset = "W13_clusterdata_10comp"
fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_data//",dataset,".dta",sep="") 
df = read.dta( fn )
df$index <- NULL

library(NbClust)
nb <- NbClust(df, distance = "euclidean", 
              min.nc=2, max.nc=15, method = "kmeans", 
              index = "alllong", alphaBeale = 0.1)
hist(nb$Best.nc[1,], breaks = max(na.omit(nb$Best.nc[1,])))