# Deep Learning for Metabolic Rate Estimation from Biosignals
Official implementation of the paper:
"Deep Learning for Metabolic Rate Estimation from Biosignals: A Comparative Study of Architectures and Signal Selection" (Babakhani, Remy, Roitberg, 2025). https://arxiv.org/abs/2511.09276


<p align="left">
  <img src="https://github.com/Sarvibabakhani/deeplearning-biosignals-ee/blob/main/figures/pipline.png"   alt="Signal pipeline" width="900"/>
  <br/>
  <em>Figure 1:  Multimodal physiological signal processing pipeline for EE. Wearable sensors placed across the body collect multimodal signals. These signals are processed and fed as input into multiple neural network architectures. (Image of sensor placement on the body is adapted from Ingraham et al. [1]).</em>
</p>

## Overview

This project investigates **deep learning methods for human metabolic rate estimation** using multimodal biosignals (e.g., heart rate, respiration, accelerometry, EMG).  
We systematically compare classical regression models with neural architectures such as **CNN, LSTM, ResNet, ResNet+Attention, and Transformers** across different sensor configurations:

- **Single signals** (e.g., minute ventilation, heart rate)  
- **Signal pairs** (e.g., heart rate + ankle acceleration)  
- **Grouped signals** (Global, Local, Hexoskin shirt signals)  

We aimed to disentangle the role of neural architecture from that of signal selection.  

**Key findings:**
- Minute ventilation is the most predictive single signal (RMSE: 0.87 W/kg with Transformer).  
- CNN and ResNet+Attention achieve strong performance for grouped or paired signals.  
- Alternatives to minute ventilation (e.g., heart rate + ankle acceleration) yield competitive results.  
- Strong **inter-individual variability** motivates adaptive and personalized modeling.

For questions, please contact: sarvenaz.babakhani@ki.uni-stuttgart.de
___
## References
[1] [Evaluating physiological signal salience for estimating metabolic energy cost from wearable sensors](https://journals.physiology.org/doi/full/10.1152/japplphysiol.00714.2018)
Kimberly A. Ingraham, Daniel P. Ferris, and C. David Remy. Journal of Applied Physiology, 126(3):717–729, 2019. doi: 10.1152/japplphysiol.00714.2018.
