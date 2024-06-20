Testing notes:

python model_CLI.py -data "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" -model all -train n -event_range "['3/18/2022 0:00','4/7/2022 23:45']" -dn 6HOURSPL_1DAYFL  -debug

C:\Users\Mikey\Documents\Github\Hysterisis-ML-Modeling\lib\ingest.py:51: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  train_range, test_range = [all_dates[0], all_dates[-1]], [all_dates[0], all_dates[-1]]
['3/18/2022 0:00', '4/7/2022 23:45'] ['Basic_LSTM', 'GRU', 'Stacked_LSTM']
predicting Basic_LSTM over ['3/18/2022 0:00', '4/7/2022 23:45']
retrieving and loading ./saved_model_multi/6HOURSPL_1DAYFL+_WSS_V/Basic_LSTM_Saved_6HOURSPL_1DAYFL+_WSS_V model.

2024-03-18 12:56:57.935947: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-18 12:56:58.847320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9436 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3080 Ti, pci bus id: 0000:01:00.0, compute capability: 8.6