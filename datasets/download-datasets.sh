#!/bin/bash

# Download the Iris dataset
[[ ! -f iris.csv ]] && \
    curl -o iris.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data && \
    echo "$(echo 'sepal_length,sepal_width,petal_length,petal_width,class'; cat iris.csv)" > iris.csv
    
[[ -f iris.csv ]] && \
    echo "Iris dataset downloaded successfully" || echo "Iris dataset download failed"
