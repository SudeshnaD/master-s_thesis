
import numpy as np
import argparse
import pickle
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


#value_slice={n:'' for n in range(10,51,10)}
n_10={k:v[:10] for k,v in aspect_dict.items()}
n_20={k:v[:20] for k,v in aspect_dict.items()}
n_30={k:v[:30] for k,v in aspect_dict.items()}
n_40={k:v[:40] for k,v in aspect_dict.items()}
n_50={k:v[:50] for k,v in aspect_dict.items()}


def codoc_freq(x,y):
    freq=0
    keywords=[x,y]
    for line in doc:
        if all(i in line for i in keywords):
            freq+=1
    return freq


def coherence_calc(dict):
    for key,values in dict.items():
        acs=0
        gs=0
        print(len(values))
        for M in range(1,len(values)):
            ls=0
            for l in range(0,M):
                try:
                    c_f=codoc_freq(values[M],values[l])
                    d_vl=vocab[values[l]]
                    ls+=np.log((c_f+1)/d_vl)    
                except ZeroDivisionError:
                    pass    
            gs+=ls
        acs+=gs
    coherence_score=acs/len(dict)
    return coherence_score

all_coh_sc={args.aspect_size:[]}

all_coh_sc[args.aspect_size].append(coherence_calc(n_10))
all_coh_sc[args.aspect_size].append(coherence_calc(n_20))
all_coh_sc[args.aspect_size].append(coherence_calc(n_30))
all_coh_sc[args.aspect_size].append(coherence_calc(n_40))
all_coh_sc[args.aspect_size].append(coherence_calc(n_50))

print(all_coh_sc)
with open('Results.txt','a') as f:
    f.write(str(all_coh_sc))