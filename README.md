# Cryptocurreny prediction using LSTM network

Project is designed to work as pipeline to easy works with various datasets and can be configurable as needed.

**Pipeline's main phases are:**
* Source - Scrapping data 
* Operation - Preparing data
* Execution - LSTM network lifecycle (building, training and verifying model)

### **Dependencies**

To run project use **Python 3.7 (64-bit version)**

Other main libraries: 

* Numpy
* Pandas
* Matplotlib
* Tensorflow
* Keras
* Scikit-learn

### **Project structure**

dataset - Datasets for given cryptos

results/graphs - Graphs create during training

results/predictions - Prediction graphs for n days

results/model - Stored trained models with weights

src/config - Config for specific stages

src/pipeline - Pipeline base code

src/execution - Execution component's code

src/operation - Operation component's code

src/source - Source component's code

