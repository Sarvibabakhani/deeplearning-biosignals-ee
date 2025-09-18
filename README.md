# Deep Learning for Metabolic Rate Estimation from Biosignals
Official implementation of the paper:
"Deep Learning for Metabolic Rate Estimation from Biosignals: A Comparative Study of Architectures and Signal Selection" (Babakhani, Remy, Roitberg, 2025).

This page is currently under construction. For questions, please contact: sarvenaz.babakhani@ki.uni-stuttgart.de

<img src="https://github.com/Sarvibabakhani/deeplearning-biosignals-ee/blob/main/figures/pipline.png"   alt="Signal pipeline" width="900"/>

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
