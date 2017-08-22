library(bnlearn)
library(foreign)
library(Rgraphviz)

## original

# dataset = "W13_voting_path_small"
# fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_data//",dataset,".dta",sep="") 
# df = read.dta( fn )
# df$index <- NULL
# 
# for(col in names(df)) {
#   df[, col] = as.factor(df[, col])
# }
# 
# ptm <- proc.time()
# res <- tabu(df) 
# plot(res)
# proc.time() - ptm
# ptm2 <- proc.time()
# 
# original = res



dataset = "W13_likeCorbynEngland"
fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_data//",dataset,".dta",sep="") 
df = read.dta( fn )
df$index <- NULL
#df$voting_path <- NULL

for(col in names(df)) {
  df[, col] = as.factor(df[, col])
}

ptm <- proc.time()
res <- tabu(df) 
plot(res)
proc.time() - ptm
ptm2 <- proc.time()

strength <- arc.strength(res, df, criterion = "loglik")
output_fn = paste("C://Users//Marios//Desktop//BES_analysis//BES_analysis_code//",dataset,"_strengths",".csv",sep="") 
write.csv(strength, output_fn )

strength.plot(res, strength, layout='dot', shape="rectangle", main = dataset)
# 'dot', 'neato', 'twopi', 'circo' and 'fdp'


# graphviz.compare(original, res)
# shd(original, res, debug = TRUE)
# unlist(compare(original,res))
# compare(target = original, current = res, arcs = TRUE)