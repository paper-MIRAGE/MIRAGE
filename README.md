# MIRAGE: Modelling Interpretable Multivariate Time Series Forecasts With Actionable Ground Explanations 

This is the official implementation of MIRAGE: Modelling Interpretable Multivariate Time Series Forecasts With Actionable Ground Explanations  _(under review at ICLR 2024)_.

<img 
    src="img\mirage_architecture.png" 
    alt="Mirage Architecture" 
    style="height:400; margin-left:auto; margin-right:auto; display:block" 
/>
  
<br>

$$ \mathcal{L}_{DPSOM}^{*} = \beta.\mathcal{L}_{SOM} + \gamma.\mathcal{L}_{Commit} + \theta.\mathcal{L}_{Reconstruction} $$

$$ \mathcal{L}_{MIRAGE} = \tau.\mathcal{L}_{Transition} + \mathcal{L}_{DPSOM} + \kappa.\mathcal{L}_{Smoothness}^{*} + \eta.\mathcal{L}_{Prediction} + \mathcal{L}_{Forecasting} $$

**Note:** Losses as a part of $ \mathcal{L}_{DPSOM}$ and the $\mathcal{L}_{Smoothness} $ are as discussed in [T-DPSOM](https://dl.acm.org/doi/10.1145/3450439.3451872)

## Abstract
Multi-variate Time Series (MTS) forecasting has made large strides (with very negligible errors) through recent advancements in neural networks, e.g., Transformers. However, in critical situations like predicting a death in an ICU or sudden gaming overindulgence; an accurate prediction without a contributing evidence is irrelevant. It is important to have model driven **Interpretability**, allowing proactive comprehension of trajectory to an extremity; and an associated **Explainability**, allowing for preventive steps; e.g., controlling BP to avoid death, or nudging players to take breaks to prevent overplay. We introduce a novel deep neural network, **MIRAGE**, which overcomes the inter-dependent challenges of (a) temporally non-smooth data trajectories for interpretability; (b) highly multi-dimensional temporal space for explainability; and (c) improving forecasting accuracy — all at once. MIRAGE: (i) achieves over **85% improvement** on the MSE of the forecasts on the most relevant SOM-VAE based SOTA networks; and (ii) unravels the intricate multi-variate relationships and temporal trajectories contributing to any sudden movement to criticalities on temporally chaotic datasets.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

In order to install and run the model, you will need a working Python 3 distribution, and optionally a NVIDIA GPU with CUDA and cuDNN installed.

## Installing

In order to install the model and run it, you have to follow these steps:

* Clone the repository, i.e. run `git clone https://github.com/paper-MIRAGE/MIRAGE`
* Change into the directory, i.e. run `cd mirage`
* Extract the data from `data` folder: `tar -xvzf players_sample_data.tar.gz`
* Install the requirements, i.e. run `pip install -r requirements.txt`
* Change into the code directory, i.e. `cd src`

## Training
After successful installation, now you should be able to run the code on our players' gameplay time-series data. The command is: `python main.py`. 


### Train on other kinds of data 

If you want to train on other types of data (eg. ETT, WTH, ECL, etc.):  
1. First save the data csv file in the `data/` folder.  
2. Run `sh scripts/data_processing.sh` according to the chosen input/output lengths.  
3. The data will be dumped as `data/model_ready_data.h5`.  
4. Make the `filename` changes in the `config.json` file.  
5. Run `python main.py`.


## Acknowledgement
We highly appreciate the following works for their valuable code and data for time series forecasting:

https://github.com/Thinklab-SJTU/Crossformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/ratschlab/dpsom

https://github.com/ratschlab/SOM-VAE