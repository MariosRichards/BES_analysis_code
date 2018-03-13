source("http://bioconductor.org/biocLite.R")
biocLite("Rgraphviz")
install.packages("foreign")
install.packages("bnlearn")
# install.packages("gRain")
# install.packages("tensorflow") # look into at some point!

library(snow)
library(parallel)
cl = makeCluster(2, type = "SOCK")
library(bnlearn)
library(foreign)
library(Rgraphviz)
# library(gRain)

rdata_dir = "C://Users//Marios//Documents//GitHub//BES_analysis//BES_analysis_data//R_data//"

filename = 'W10_onlyall_red_ordinal_vars'
fn = paste(rdata_dir,filename,".dta",sep="") 
df = read.dta( fn )
df$index <- NULL
# set NA values to -1
df[is.na(df)] = -1
# turn everything into categories
for(col in names(df)) {
  df[, col] = as.factor(df[, col])
}  

filename = 'W10_onlycore_values'
fn = paste(rdata_dir,filename,".dta",sep="") 
df2 = read.dta( fn )
df2$index <- NULL
df2[is.na(df2)] = -1
# turn everything into categories
for(col in names(df2)) {
  df2[, col] = as.factor(df2[, col])
}

# dedup(df)

filename = 'W10_onlymore_core_values'
fn = paste(rdata_dir,filename,".dta",sep="") 
df2 = read.dta( fn )
df2$index <- NULL
df2[is.na(df2)] = -1
# turn everything into categories
for(col in names(df2)) {
  df2[, col] = as.factor(df2[, col])
}



# for(col in names(df)) {
#   df[, col] = as.numeric(df[, col])
# }

#res <- hc(df) # 3 mins
#res <- rsmax2(df) # 3 mins
#res <- aracne(df) # 3 mins

ptm <- proc.time()
res <- tabu(df) # 3 mins
plot(res)
proc.time() - ptm
ptm2 <- proc.time()
ptm2-ptm

strength <- arc.strength(res, df, criterion = "loglik")
#strength <- arc.strength(res, df, criterion = "bic")

strength.plot( res, strength, layout="dot", shape="rectangle")
#strength.plot( res, strength, layout="neato", shape="rectangle")


output_fn = paste(rdata_dir,filename,"_strengths",".csv",sep="") 
write.csv(strength, output_fn )



# time out 500 via 5
# R= 2 -> 15 mins, R=30 -> 6.4 hrs
# parallel R=50 -> 17436 4.8 hrs
ptm <- proc.time() 
boot3 <- boot.strength(df, R = 50, algorithm = "tabu",
                      algorithm.args = list(score = "bde", iss = 10), cluster=cl)
ptm2 <- proc.time()
ptm2-ptm

boot3[(boot3$strength > 0.95) & (boot3$direction >= 0.5), ]
plot(boot$direction)

avg.boot = averaged.network(boot3, threshold = 0.95)
graphviz.plot(avg.boot, layout="dot", shape="rectangle" )


# R=10 -> 46.77, R=20 -> 94, R=500 -> 2302, you could do ~5,000 overnight
ptm <- proc.time()
boot4 <- boot.strength(df2, R = 500, algorithm = "tabu",
                       algorithm.args = list(score = "bde", iss = 10), cluster=cl)
ptm2 <- proc.time()
ptm2-ptm

boot4[(boot4$strength > 0.95) & (boot4$direction >= 0.5), ]
avg.boot = averaged.network(boot4, threshold = 0.95)
# graphviz.plot(avg.boot, layout="dot", shape="rectangle" )

strength <- arc.strength(avg.boot, df2, criterion = "bde")
strength.plot( avg.boot, strength, layout="dot", shape="rectangle")

# parallel clusters R=20 -> 90 - so 1000 -> bit over an hour
# 
ptm <- proc.time()
boot4 <- boot.strength(df2, R = 20, algorithm = "tabu",
                       algorithm.args = list(score = "bde", iss = 10), cluster=cl)
ptm2 <- proc.time()
ptm2-ptm
