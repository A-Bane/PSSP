Here are the changes you need to make before running the code:

1. In dataload.py change directory name in line 7 accordingly.

2. For running Q8 prediction:
2.1.In main.py line 32, change "label_train = hot2labelQ3(label_train)" to "label_train = hot2labelQ8(label_train)" for Q8 prediction.
2.2. Similarly, line 52, change "label_test = hot2labelQ3(label_test)" to "label_test = hot2labelQ8(label_test)" for Q8 prediction.
2.3. Change --q default value to 8
2.4. Change --out_class_number default value to 9

3. For running Q3 prediction:
2.1.In main.py line 32, change to "label_train = hot2labelQ3(label_train)"
2.2. Similarly, line 52, change to "label_test = hot2labelQ3(label_test)"
2.3. Change --q default value to 3
2.4. Change --out_class_number default value to 4

4. You can also alter other hyperparameters from 116-131 line in the parser.