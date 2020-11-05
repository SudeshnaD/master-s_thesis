# Deep Learning Approaches for Opinion Mining on Conversational Social Media Texts

## Abstract
Opinions and reviews help individuals to make decisions. Opinion mining on social media conversational text such as forum discussions, product reviews provides valuable insight on user sentiment. In a span of review text, opinions are expressed towards aspect terms. To understand the fine-grained sentiment expressed toward each aspect in a product review, aspect term identification and aspect term categorization is needed. Jointly they are referred to as aspect extraction. In this thesis, we perform an empirical study of three unsupervised deep learning models to perform aspect extraction on the Sentihood dataset. To evaluate the models we use multiple word embeddings to conduct extensive experiments. We evaluate our observations using quantitative and qualitative analysis of extracted aspects, and the micro F1 scores for the task of aspect classification.

## Environment
- python 3.7.1 
- keras 2.3.1
- tensorflow 2.0.0
- scikit-learn 0.22.1
- pandas 1.0.3
- theano 1.0.4
- mittens 0.2
- gensim 3.8.0


## Folder Structure

#### ABAE
- code: scripts for implementing the ABAE model.
- preprocessed_data: cleaned dataset for experiments.
- CoherenceScore: script for calculating UMass coherence score and WETC score, and results for ABAE and ABAE_SH model.

#### AE_Sememes
- AE-SA : code for implementing the AE-SA model.
- AE-CSA : code for implementing the AE-CSA model.

#### Baseline models
Code for implementing NMF and Kmeans baseline models.

#### Coherence_Visualization
Notebooks to visualize and calculate area under curve (AUC) for the coherence scores for all models.

#### Literature
Papers referred for the thesis.

#### Report
The thesis report.
