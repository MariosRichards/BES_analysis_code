source("http://bioconductor.org/biocLite.R")
biocLite("Rgraphviz")
install.packages("foreign")
install.packages("bnlearn")

library(bnlearn)
library(foreign)
library(Rgraphviz)

rdata_dir = "C://Users//Marios//Documents//GitHub//BES_analysis//BES_analysis_data//R_data//"

filename = 'W13_onlycore_values_with_nans'
fn = paste(rdata_dir,filename,".dta",sep="") 
df = read.dta( fn )
df$index <- NULL



dataset = "W13_reduced_with_nans"
fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_data//",dataset,".dta",sep="") 
df = read.dta( fn )
df$index <- NULL

for(col in names(df)) {
  #  print(fac)
  df[, col] = as.factor(df[, col])
}



df[is.na(df)] = -1

# load factor list
factor_list = "W13_reduced_with_nans_factor_variables"

fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_data//",factor_list,".dta",sep="") 
fl = read.dta( fn )
for(fac in fl$index) {
#  print(fac)
  df[, fac] = as.factor(df[, fac])
 # df$fac = factor(df$fac) 
}


# now using larger "reduced" dataset
dataset = "W13_reduced"
fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_data//",dataset,".dta",sep="") 
df = read.dta( fn )
df$index <- NULL




# 10:5s, 20:16s, 30: 80s, 40: 195.45s, 50: 369.05
# with no missing values!

# quicker with
# debug=TRUE

# boo! Too slow!

ptm <- proc.time()
res <- tabu(df) 
plot(res)
proc.time() - ptm
ptm2 <- proc.time()

# problem with using mixed continuous/discrete dataset approach
# continuous -> discrete links are rendered illegal!



# negative loglikelihoods!

ptm <- proc.time()
arc.strength(res, df, criterion = "loglik-cg")
proc.time() - ptm


# less helpful, because everything (that makes it over the threshold) is v v small!

ptm <- proc.time()
strength <- arc.strength(res, df, criterion = "mi-cg", debug=TRUE)
proc.time() - ptm


#res <- set.arc(res) # 
strength <- arc.strength(res, df, criterion = "loglik")
#strength <- arc.strength(res, df, criterion = "mi-cg")
output_fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_code//",dataset,"_strengths",".csv",sep="") 
write.csv(strength, output_fn )

# pain in the ass
# won't run unless there exist *complete* rows

# I think we need the updated version of R to install this function ...

ptm = proc.time()
# df[is.na(df)] =0
res = structural.em(df[5])
plot(res)
proc.time() - ptm
ptm2 = proc.time()


# data is being treated as if continuous!


hl_var = "leftRight__Right"
# bleh still hard to make out
# strength.plot(res, strength, layout="dot", shape="rectangle") 
strength.plot( res, strength, layout="dot", shape="rectangle")
               
               #, mb(res, hl_var))  )


# df$generalElectionVote = factor(df$generalElectionVote)

# arc.strength(res, df[0:50], criterion = "loglik-g")
fitted = bn.fit(res, df)
net = bn.net(fitted)
graphviz.plot(net, layout="dot", shape="rectangle")

