library(keras)
library(tidytext)
library(tidyverse)

text_data <- read.csv('scripts.csv')

text_data <- text_data  %>%
  filter(grepl('jerry',tolower(Character))) %>%
  mutate(Dialogue=stringr::str_replace_all(Dialogue, "[[:punct:]]", " ")) %>%
  filter(Season<=7) %>%
  select(Dialogue) %>% mutate(Dialogue= tolower(as.character(Dialogue))) %>% pull()

text_data <- paste(text_data,collapse='')

text_df <- data.frame(text=text_data)

text_to_num <- text_df %>% mutate(text = as.character(text)) %>% unnest_tokens(word,text) %>% unique() %>% arrange(desc(word))
text_to_num$index <- 1:(dim(text_to_num)[1])

encoded <- text_df %>%
  mutate(text=as.character(text)) %>%
  unnest_tokens(word,text) %>%
  inner_join(text_to_num,by=c('word'))

vocab_size = length(text_to_num$word)+1
cat("Vocab Size",vocab_size)


sequences <- text_df %>%
  mutate(text=as.character(text)) %>%
  unnest_tokens(ngram,text,token='ngrams',n=2) %>%
  tidyr::separate(ngram,into=c('word1','word2'),sep=" ") %>%
  rename('word'='word1') %>%
  inner_join(text_to_num,by=c('word')) %>%
  select(-word) %>%
  rename(word1=index,word=word2) %>%
  inner_join(text_to_num,by=c('word')) %>%
  select(-word) %>%
  rename(word2=index)


X <- as.numeric(sequences$word1)
y <- as.numeric(sequences$word2)
y <- keras::to_categorical(y,num_classes = vocab_size )

model <- keras::keras_model_sequential()
model %>%
  layer_embedding(vocab_size,10,input_length=1) %>%
  layer_lstm(50) %>%
  
  
  
  layer_dense(vocab_size,activation='softmax')
summary(model)

optimizer <- optimizer_adam(lr = 0.01)

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)
# fit network

model %>% fit(
  X, y,
  epochs = 500
)


generate_seq <- function(model,text_to_num,seed_text,n_words){
  
  in_text = seed_text
  result = seed_text
  for(i in 1:n_words){
    
    encoded_num <- as.vector(text_to_num %>% filter(word==in_text) %>% select(index) %>% pull())
    yhat = model %>% predict_classes(encoded_num, verbose=2)
    yhat <- as.numeric(yhat)
    
    out_word = ''
    for (j in text_to_num$index){
      if(j==yhat){
        text_to_print <- text_to_num %>% filter(index==yhat) %>% select(word) %>% pull()
        out_word = text_to_print
        
      }
    }
    in_text = out_word
    result = stringr::str_c(result,out_word,sep=" ")
    
    
  }
  return(result)
}



print(generate_seq(model,text_to_num,'you',6))

