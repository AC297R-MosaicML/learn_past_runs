# Improving Deep Learning Model Over Multiple Runs
## Introduction

One of the primary bottlenecks in machine learning pipelines is the time required to train the model; consequently, any innovations that improve time-efficiency in the training process have the potential to greatly expand the utility of machine learning as a tool across a broad range of applications. In particular, this project focuses on leveraging data from prior training runs to decrease the time required for subsequent model re-trainings, e.g. on newly gathered data.	


## How to run

```
git clone https://github.com/AC297R-MosaicML/learn_past_runs
cd learn_past_runs
chmod +x run.sh && ./run.sh
```


## File Structure
- experiment.py: highest-level, call train and eval
- train.py: training 
- eval.py: validating
- models.py: contains models
- dataset.py: dataset and pre-processing
- metrics.py: evaluation metrics
- plotting.py: graphing utilities
