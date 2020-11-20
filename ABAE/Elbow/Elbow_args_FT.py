
import subprocess
import logging
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-s", type=int, metavar='<int>', default=100,
                    help="Starting asp count")
parser.add_argument("-e", type=int, metavar='<int>', default=100,
                    help="Ending asp count")
args=parser.parse_args()

ind_clus_sim={}
cluster_sim={}
#aspect_wise_sim=[]
#mean_sim=[]


""" def elbow(cos_sim_g):
    try:
        #avg = [float(sum(col))/len(col) for col in zip(*cos_sim)]
        avg = [float(sum(col))/len(col) for col in cos_sim_g]
        mean_avg=sum(avg)/len(avg)
        stats=(avg,mean_avg)
        #with open('avg_meanavg.txt','w') as f:
        #    f.write(str(stats))
    except Exception as e:
        print(e)
    return stats """

def listoflist(s):
    s=s.lstrip('[[')
    s=s.rstrip(']]')
    s=s.split('], [')
    xs=[x.split(', ') for x in s]
    def str_flt(l):
        l=[float(x) for x in l]
        return l
    lol=[]
    for x in xs:
        fltx=str_flt(x)
        lol.append(fltx)
    #lol.append([str_flt(x) for x in xs])
    return lol



def elbow(K):
    try:
        with open('FTcos_sim_g{}.txt'.format(str(K)),'r') as s:
            doc=s.readlines()
        doclist=listoflist(doc[0])

        avg = [float(sum(col))/len(col) for col in doclist]
        mean_avg=sum(avg)/len(avg)
        stats=(avg,mean_avg)
        with open('FTavg_meanavg{}{}.txt'.format(args.s,args.e),'a') as f:
            f.write('{}: '.format(K))
            f.write(str(stats))
    except Exception as e:
        print(e)
    return stats

""" if __name__=='__main__':
    K=50
    elbow(K) """

if __name__=='__main__':
    with open('FTavg_meanavg{}{}.txt'.format(args.s,args.e),'w') as f:
            pass
    for K in range(args.s,args.e):
        #cmd=['python','train.py', '--emb-name', '../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec', '--epochs','15', '--domain','sentihood', '--out-dir','output_dir','-aspect_size']
        cmd=['python','trainFT.py', '--emb-name', '../preprocessed_data/sentihood/flat_repoglove.txt.word2vec', '--epochs','15', '--domain','sentihood', '--out-dir','output_dir','-aspect_size']
        cmd.append(str(K))
        subprocess.check_output(cmd)
        elbow(K)



""" if __name__=='__main__':

      
    #cmd = 'python train.py --emb-name ../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec --epochs 1 --domain sentihood --out-dir output_dir'

    for K in range(args.s,args.e):
        #try:
            print('for K {}'.format(K))
            #cmd=['python','elbow_tc1.py']
            cmd=['python','train.py', '--emb-name', '../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec', '--epochs','15', '--domain','sentihood', '--out-dir','output_dir','-aspect_size']
            cmd.append(str(K))
            p = subprocess.check_output(cmd)
            output=str(p,'utf-8')
            print(output)
            #ot=output.split('], ')[1].rstrip(')\r\n')
            #cluster_sim[K]=float(ot[0])
            #otlist=[float(i) for i in ot[0].split(', ')]
            #ot_2=output.split('], ')[0].lstrip('Creating vocab ...\n   30265 total words, 3793 unique words\n  keep the top 9000 words\n Reading dataset ...\n  train set\n   <num> hit rate: 0.85%, <unk> hit rate: 0.00%\n  test set\n   <num> hit rate: 0.51%, <unk> hit rate: 5.13%\nNumber of training examples:  3724\n69 69 69\nLength of vocab:  3796\nin create model  69\nLoading embeddings from: ../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec')
            #ot_2=ot_2.lstrip('([')
            #otlist=[float(i) for i in ot_2.split(', ')]
            #print(ot)
            #ind_clus_sim[K]=[float(i) for i in ot_2.split(', ')]
        #except Exception as e:
        #    print(e)
        #    continue

        

    # with open('cluster_sim.txt','w') as f:
    #     f.write(str(cluster_sim))

    # with open('ind_cluster_sim.txt','w') as f:
    #     f.write(str(ind_clus_sim)) """

"""     result = out.split('\n')
    for lin in result:
        if not lin.startswith('#'):
            print(lin) """