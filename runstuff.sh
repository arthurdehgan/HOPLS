for ss in 10 20
do
	for snr in 5 0 -2 -5
	do
		for ym in 2 3
		do
			python pipeline.py simple $ss 3 $ym $snr
		done
	done
done
