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