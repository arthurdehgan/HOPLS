for ss in 10 20
do
	for snr in 5 0 -2 -5
	do
		for nm in simple complex
		do
			python pipeline.py $nm $ss 5 2 $snr
		done
	done
done
