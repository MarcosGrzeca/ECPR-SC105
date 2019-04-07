#http://pablobarbera.com/ECPR-SC105/code/16-word-embeddings.html

#install.packages("devtools")
# library(devtools)
# install_github("mukul13/rword2vec")
library(rword2vec)
library(lsa)
library(readr)
library(quanteda)

fb <- read.csv("data/incivility.csv", stringsAsFactors = FALSE)
fbcorpus <- corpus(fb$comment_message)
fbdfm <- dfm(fbcorpus, remove=stopwords("english"), verbose=TRUE)
fbdfm <- dfm_trim(fbdfm, min_docfreq = 2, verbose=TRUE)

#bin_to_txt("FBvec.bin", "FBvector.txt")

# extracting word embeddings for words in corpus
w2v <- readr::read_delim("code/FBvector.txt", 
                  skip=1, delim=" ", quote="",
                  col_names=c("word", paste0("V", 1:100)))

w2v <- w2v[w2v$word %in% featnames(fbdfm),]

# creating new feature matrix for embeddings
embed <- matrix(NA, nrow=ndoc(fbdfm), ncol=100)
for (i in 1:ndoc(fbdfm)){
  if (i %% 100 == 0) message(i, '/', ndoc(fbdfm))
  # extract word counts
  vec <- as.numeric(fbdfm[i,])
  # keep words with counts of 1 or more
  doc_words <- featnames(fbdfm)[vec>0]
  # extract embeddings for those words
  embed_vec <- w2v[w2v$word %in% doc_words, 2:101]
  # aggregate from word- to document-level embeddings by taking AVG
  embed[i,] <- colMeans(embed_vec, na.rm=TRUE)
  # if no words in embeddings, simply set to 0
  if (nrow(embed_vec)==0) embed[i,] <- 0
}

set.seed(123)
training <- sample(1:nrow(fb), floor(.80 * nrow(fb)))
test <- (1:nrow(fb))[1:nrow(fb) %in% training == FALSE]

## function to compute accuracy
accuracy <- function(ypred, y){
    tab <- table(ypred, y)
    return(sum(diag(tab))/sum(tab))
}
# function to compute precision
precision <- function(ypred, y){
    tab <- table(ypred, y)
    return((tab[2,2])/(tab[2,1]+tab[2,2]))
}
# function to compute recall
recall <- function(ypred, y){
    tab <- table(ypred, y)
    return(tab[2,2]/(tab[1,2]+tab[2,2]))
}

library(glmnet)
require(doMC)
registerDoMC(cores=3)
# confusion matrix
# table(preds, fb$attacks[test])

# performance metrics
# accuracy(preds, fb$attacks[test])
# precision(preds==1, fb$attacks[test]==1)
# recall(preds==1, fb$attacks[test]==1)
#precision(preds==0, fb$attacks[test]==0)
#recall(preds==0, fb$attacks[test]==0)

## identifying predictive features
# df <- data.frame(coef = as.numeric(beta),
#                 word = names(beta), stringsAsFactors=F)
# df <- df[order(df$coef),]
# head(df[,c("coef", "word")], n=30)
# df <- df[order(df$coef, decreasing=TRUE),]
# head(df[,c("coef", "word")], n=30)
#head(w2v[order(w2v$V83, decreasing=TRUE),"word"], n=20)

library(xgboost)
# converting matrix object
X <- as(cbind(fbdfm, embed), "dgCMatrix")
# parameters to explore
tryEta <- c(1,2)
tryDepths <- c(1,2,4)
# placeholders for now
bestEta=NA
bestDepth=NA
bestAcc=0

for(eta in tryEta){
  for(dp in tryDepths){ 
    bst <- xgb.cv(data = X[training,], 
            label =  fb$attacks[training], 
            max.depth = dp,
          eta = eta, 
          nthread = 4,
          nround = 500,
          nfold=5,
          print_every_n = 100L,
          objective = "binary:logistic")
    # cross-validated accuracy
    acc <- 1-mean(tail(bst$evaluation_log$test_error_mean))
        cat("Results for eta=",eta," and depth=", dp, " : ",
                acc," accuracy.\n",sep="")
        if(acc>bestAcc){
                bestEta=eta
                bestAcc=acc
                bestDepth=dp
        }
    }
}

cat("Best model has eta=",bestEta," and depth=", bestDepth, " : ",
    bestAcc," accuracy.\n",sep="")

# running best model
rf <- xgboost(data = X[training,], 
    label = fb$attacks[training], 
        max.depth = bestDepth,
    eta = bestEta, 
    nthread = 4,
    nround = 1000,
        print_every_n=100L,
    objective = "binary:logistic")

preds <- predict(rf, X[test,])
cat("\nAccuracy on test set=", round(accuracy(preds>.50, fb$attacks[test]),3))
cat("\nPrecision(1) on test set=", round(precision(preds>.50, fb$attacks[test]),3))
cat("\nRecall(1) on test set=", round(recall(preds>.50, fb$attacks[test]),3))
# cat("\nPrecision(0) on test set=", round(precision(preds<.50, fb$attacks[test]==0),3))
# cat("\nRecall(0) on test set=", round(recall(preds<.50, fb$attacks[test]==0),3))
