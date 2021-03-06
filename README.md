# Deep Learning Approaches for Opinion Mining on Conversational Social Media Texts

The [thesis report](https://github.com/SudeshnaD/master-s_thesis/blob/main/Report/M_Info_15102020_Dasgupta_Sudeshna.pdf) and [final presentation](https://github.com/SudeshnaD/master-s_thesis/blob/main/Report/Thesis_final%20Presentation_SudeshnaDasgupta.pdf).

## Abstract
Opinions and reviews help individuals to make decisions. Opinion mining on social media conversational text such as forum discussions, product reviews provides valuable insight on user sentiment. In a span of review text, opinions are expressed towards aspect terms. To understand the fine-grained sentiment expressed toward each aspect in a product review, aspect term identification and aspect term categorization is needed. Jointly they are referred to as **aspect extraction**. In this thesis, we perform an empirical study of three unsupervised deep learning models to perform aspect extraction on the Sentihood dataset. 
>To evaluate the models we use multiple word embeddings to conduct extensive experiments. We evaluate our observations using quantitative and qualitative analysis of extracted aspects, and the micro F1 scores for the task of aspect classification.

## Research Objective

*Empirical study of unsupervised aspect extraction on Sentihood dataset.*

**Research questions:** 
1. Underlying challenges in extracting aspects from a dataset with
	 - large number of aspect labels
	 - explicit as well as implicit aspect labels 
2. If domain of dataset influences the choice of approach for aspect extraction 
3. if transfer learning can resolve the inadequacy of training data.

## Models implemented and evaluated: 
- Attention Based Aspect Extraction (ABAE) 
- Aspect Extraction with Sememe Attentions (AE-SA)
- Aspect Extraction via Context-enhanced Sememe Attentions (AE-CSA)

#### Sentihood dataset: 
- Training corpus: 3724 samples
- Test dataset: 1003 samples
- Dataset consists of reviews about the urban neighborhoods in London. 
- 12 predefined aspect labels:
 ’live’, ’safety’, ’price’, ’quiet’, ’dining’, ’nightlife’, ’transit-location’, ’touristy’, ’shopping’, ’green-culture’, ’multicultural’, ’general’


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

**ABAE**
- code: scripts for implementing the ABAE model, forked from  https://github.com/madrugado/Attention-Based-Aspect-Extraction
- preprocessed_data: 
	- cleaned dataset for experiments
	- pre-trained word embeddings.
- CoherenceScore: 
	- script for calculating UMass coherence score and WETC score
	- results for ABAE and ABAE_SH model.

**AE_Sememes**
- AE-SA : code for implementing the AE-SA model.
- AE-CSA : code for implementing the AE-CSA model.


**Baseline models** : Code for implementing NMF and Kmeans baseline models.

**Coherence_Visualization** : Notebooks to visualize and calculate area under curve (AUC) for the coherence scores for all models.

**Literature** : Papers referred for the thesis.

**Report** : The thesis report and final presentation.

## For training:

Run in code/, code_aesa/, code_aecsa/ folder:

```python train.py --emb-name ../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec --epochs 1 --domain sentihood --out-dir output_dir -as $x```

where *emb-name* is the path to the pre-trained word embeddings, *out-dir* is the path of the output directory and as is the number of inferred aspects to be generated. 

After training, two output files will be saved in *code/output_dir/* , which are
1) *aspect.log* :  list of extracted aspects as top 100 words for each aspect
2) *model_param* contains the saved model weights.

## Evaluation
We rebuild the network architecture and then load the saved model weights. Values of arguments used are the same as those for training.
Under code/ and type the following command:
```
python evaluation_sent.py \
--domain sentihood \
-o output_dir 
```

This will output a file *att_weights* that contains the attention weights on all test sentences in code/output_dir/.

To assign each test sentence a gold aspect label, a cluster map is created by manually mapping each inferred aspect to a gold aspect label according to its top words. For evaluaton using F scores, the trained model is used as sentence classifier where predicted class of a sentence is a gold aspect label.
