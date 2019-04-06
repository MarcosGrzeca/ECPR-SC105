#http://pablobarbera.com/ECPR-SC105/code/16-word-embeddings.html

#install.packages("devtools")
# library(devtools)
# install_github("mukul13/rword2vec")

library(rword2vec)
library(lsa)

distance(file_name = "code/vec.bin",
        search_word = "princess",
        num = 10)

library(readr)
data <- read_delim("code/vector.txt", 
    skip=1, delim=" ",
    col_names=c("word", paste0("V", 1:100)))

plot_words <- function(words, data){
  # empty plot
  plot(0, 0, xlim=c(-2.5, 2.5), ylim=c(-2.5,2.5), type="n",
       xlab="First dimension", ylab="Second dimension")
  for (word in words){
    # extract first two dimensions
    vector <- as.numeric(data[data$word==word,2:3])
    # add to plot
    text(vector[1], vector[2], labels=word)
  }
}

plot_words(c("good", "better", "bad", "worse"), data)

similarity <- function(word1, word2){
    lsa::cosine(
        x=as.numeric(data[data$word==word1,2:101]),
        y=as.numeric(data[data$word==word2,2:101]))

}

similarity("australia", "england")

word_analogy(file_name = "code/vec.bin",
    search_words = "king queen man",
    num = 1)

word_analogy(file_name = "code/vec.bin",
    search_words = "paris france berlin",
    num = 1)

distance(file_name = "code/FBvec.bin",
        search_word = "liberal",
        num = 10)

library(quanteda)

fb <- read.csv("data/incivility.csv", stringsAsFactors = FALSE)
fbcorpus <- corpus(fb$comment_message)
fbdfm <- dfm(fbcorpus, remove=stopwords("english"), verbose=TRUE)
fbdfm <- dfm_trim(fbdfm, min_docfreq = 2, verbose=TRUE)

#bin_to_txt("code/FBvec.bin", "FBvector.txt")

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