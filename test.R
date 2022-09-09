library(keras)
library(tensorflow)

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 5

data_augmentation <- FALSE

# Reproducibility
set_random_seed(12345)

# Data Preparation --------------------------------------------------------

# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x / 255
y_train <- cifar10$train$y
y_test <- cifar10$test$y



compute <- function(idx, x_train, y_train, x_test, y_test) {
  # Initialize sequential model
  model <- keras_model_sequential()
  
  # # Defining Model ----------------------------------------------------------
  #
  # if (data_augmentation) {
  #   data_augmentation = keras_model_sequential() %>%
  #     layer_random_flip("horizontal") %>%
  #     layer_random_rotation(0.2)
  #
  #   model <- model %>%
  #     data_augmentation()
  # }
  
  model <- model %>%
    # Start with hidden 2D convolutional layer being fed 32x32 pixel images
    layer_conv_2d(
      filter = 16, kernel_size = c(3,3), padding = "same", 
      input_shape = c(32, 32, 3)
    ) %>%
    layer_activation_leaky_relu(0.1) %>% 
    
    # Second hidden layer
    layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
    layer_activation_leaky_relu(0.1) %>% 
    
    # Use max pooling
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    # 2 additional hidden 2D convolutional layers
    layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
    layer_activation_leaky_relu(0.1) %>% 
    layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
    layer_activation_leaky_relu(0.1) %>% 
    
    # Use max pooling once more
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    # Flatten max filtered output into feature vector 
    # and feed into dense layer
    layer_flatten() %>%
    layer_dense(256) %>%
    layer_activation_leaky_relu(0.1) %>% 
    layer_dropout(0.5) %>%
    
    # Outputs from dense layer are projected onto 10 unit output layer
    layer_dense(10)
  
  opt <-
    optimizer_adamax(
      learning_rate = learning_rate_schedule_exponential_decay(
        initial_learning_rate = 5e-3,
        decay_rate = 0.96,
        decay_steps = 1500,
        staircase = TRUE
      )
    )
  
  model %>% compile(
    loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
    optimizer = opt,
    metrics = "accuracy"
  )
  
  
  # Training ----------------------------------------------------------------
  model %>% fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, y_test),
    shuffle = FALSE,
    verbose = 0
  )
  
  score <- model %>% evaluate(x_test, y_test, verbose = 0)
  
  # Result
  data.frame(idx = idx,
             loss = score["loss"],
             epochs = epochs)
  
}


# DoMC + foreach
library(foreach)
library(doMC)
registerDoMC(cores = 2)

res <- foreach(i = 1:4, .combine = "rbind") %dopar% {

  x_train_ind <- sample(dim(x_train)[1], 1000)
  x_test_ind <- sample(dim(x_test)[1], 50)
  
  y_test_ind <- sample(dim(y_test)[1], 50)
  y_train_ind <- sample(dim(y_train)[1], 1000)
  
  compute(i,
          x_train[x_train_ind, 1:32, 1:32, 1:3],
          y_train[y_train_ind, 1],
          x_test[x_test_ind, 1:32, 1:32, 1:3],
          y_test[y_test_ind, 1])
  
}

res


# ClusterMQ + foreach
options(clustermq.scheduler = "multicore")
library(clustermq)
library(foreach)

register_dopar_cmq(n_jobs = 2, memory = 10240)

res <- foreach(i = 1:4, .combine = "rbind") %dopar% {

  x_train_ind <- sample(dim(x_train)[1], 1000)
  x_test_ind <- sample(dim(x_test)[1], 50)
  
  y_test_ind <- sample(dim(y_test)[1], 50)
  y_train_ind <- sample(dim(y_train)[1], 1000)
  
  compute(i, 
          x_train[x_train_ind, 1:32, 1:32, 1:3],
          y_train[y_train_ind, 1],
          x_test[x_test_ind, 1:32, 1:32, 1:3],
          y_test[y_test_ind, 1])
}

res
