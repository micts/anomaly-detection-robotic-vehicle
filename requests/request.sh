curl -i -H "Content-Type: application/json" -X POST \
	-d '{"CPU": 26.0, 
             "RxKBTot": 0.16, 
	     "TxKBTot": 0.0, 
	     "WriteKBTot": 1.68, 
	     "RMS": 0.051649, 
	     "diff_encoder_l": 25.034476, 
	     "Volts": 203.633387, 
             "R/T(xKBTot)": 1.16}' 127.0.0.1:5000/predict
