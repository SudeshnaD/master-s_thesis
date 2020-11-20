for k in 10 20 30 40 50
do
python evaluation_sent.py --domain sentihood --out-dir output_dir/glove_SentHd_50itr --aspect_size $k
done