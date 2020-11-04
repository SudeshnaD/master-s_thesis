
import numpy as np
import argparse
import pickle
import os
from w2vEmbReader import W2VEmbReader as EmbReader
from Aspect_parse import aspect_parse
parser = argparse.ArgumentParser()
parser.add_argument("-as", "--aspect_size", dest="aspect_size", type=int, metavar='<int>', default=10,
                    help="aspect_size")
args = parser.parse_args()


aspect_dict=aspect_parse(args.aspect_size)


with open('train.txt','r') as s:
    doc=s.readlines()
    doc=[t.strip('\n') for t in doc]


with open("vocab_c.pkl","rb") as f:
    vocab=pickle.load(f)

emb_name='../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec'
emb_reader = EmbReader(os.path.join("..", "preprocessed_data"), emb_name)

#value_slice={n:'' for n in range(10,51,10)}
n_10={k:v[:10] for k,v in aspect_dict.items()}
n_20={k:v[:20] for k,v in aspect_dict.items()}
n_30={k:v[:30] for k,v in aspect_dict.items()}
n_40={k:v[:40] for k,v in aspect_dict.items()}
n_50={k:v[:50] for k,v in aspect_dict.items()}


def codoc_freq(x,y):
    xe=emb_reader.get_emb_given_word(x)
    ye=emb_reader.get_emb_given_word(y)
    return np.inner(xe,ye)


def coherence_calc(dict):
    acs=0
    for key,values in dict.items():
        gs=0
        print(len(values))
        print(key)
        for j in range(1,len(values)):
            ls=0
            for i in range(0,j):
                try:
                    c_f=codoc_freq(values[i],values[j])
                    ls+=c_f
                except ZeroDivisionError:
                    pass
                except Exception as e:
                    print(i,j)    
            gs+=ls
            N=len(values)
            gs=gs/(N*(N-1))
        acs+=gs
    coherence_score=acs/len(dict)
    return coherence_score

print(coherence_calc(n_10))

#all_coh_sc={args.aspect_size:[]}

#all_coh_sc[args.aspect_size].append(coherence_calc(n_10))
#all_coh_sc[args.aspect_size].append(coherence_calc(n_20))
#all_coh_sc[args.aspect_size].append(coherence_calc(n_30))
#all_coh_sc[args.aspect_size].append(coherence_calc(n_40))
#all_coh_sc[args.aspect_size].append(coherence_calc(n_50))

#with open('Results.txt','a') as f:
#    f.write(str(all_coh_sc))

#print(all_coh_sc)