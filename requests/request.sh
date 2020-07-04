curl -i -H "Content-Type: application/json" -X POST \
	-d '{"CPU": 27.26, 
             "RxKBTot": 1.74, 
	     "TxKBTot": 1.26, 
	     "WriteKBTot": 0.0, 
	     "RMS": 0.433838, 
	     "diff_encoder_l": 12.354838, 
	     "Volts": 156.025235, 
             "R/T(xKBTot)": 1.2123}' 127.0.0.1:5000/predict
