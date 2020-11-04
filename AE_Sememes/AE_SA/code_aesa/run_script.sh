for x in range(55,90,5):

do
    python train.py --emb-name ../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec --epochs 1 --domain sentihood --out-dir output_dir -as $x

done