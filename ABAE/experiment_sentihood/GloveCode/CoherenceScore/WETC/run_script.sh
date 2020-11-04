for k in 60 70 80 90 100
do
	python wetc_score.py --aspect_size $k 1>out_log.txt
done