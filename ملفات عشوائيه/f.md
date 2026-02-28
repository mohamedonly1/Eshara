Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Install the latest PowerShell for new features and improvements! https://aka.ms/PSWindows

PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python collect_data.py
C:\Users\`mohamedonly1\AppData\Local\Programs\Python\Python313\python.exe: can't open file 'C:\\Users\\`mohamedonly1\\Desktop\\hand-gesture-recognition-mediapipe-main\\collect_data.py': [Errno 2] No such file or directory
PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> venv310\Scripts\sctivate
venv310\Scripts\sctivate : The module 'venv310' could not be loaded. For more information, run 'Import-Module venv310'.
At line:1 char:1
+ venv310\Scripts\sctivate
+ ~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (venv310\Scripts\sctivate:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CouldNotAutoLoadModule

PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> venv310\Scripts\activate
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python train_model.py
==================================================
  تدريب موديل لغة الإشارة العربية
==================================================

📂 الحروف المتاحة: 29
✅ تم تحميل 2044 عينة
📊 توزيع العينات:
  ✅ أ: 201 ████████████████████
  ✅ ب: 208 ████████████████████
  ✅ ت: 201 ████████████████████
  ✅ ث: 216 █████████████████████
  ✅ ج: 204 ████████████████████
  ✅ ح: 202 ████████████████████
  ✅ خ: 202 ████████████████████
  ✅ د: 200 ████████████████████
  ✅ ذ: 208 ████████████████████
  ✅ ر: 202 ████████████████████

🔢 عدد الفئات: 10
✅ بيانات التدريب: 1635
✅ بيانات الاختبار: 409
2026-02-26 23:23:05.439761: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:23:05.440157: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

🏗️  بناء الموديل...
2026-02-26 23:23:14.541233: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:23:14.548793: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
2026-02-26 23:23:14.563731: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublasLt64_11.dll'; dlerror: cublasLt64_11.dll not found
2026-02-26 23:23:14.569885: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2026-02-26 23:23:14.584664: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2026-02-26 23:23:14.596349: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusolver64_11.dll'; dlerror: cusolver64_11.dll not found
2026-02-26 23:23:14.607898: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusparse64_11.dll'; dlerror: cusparse64_11.dll not found
2026-02-26 23:23:14.615964: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
2026-02-26 23:23:14.616627: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2026-02-26 23:23:14.621850: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 128)               5504

 batch_normalization (BatchN  (None, 128)              512
 ormalization)

 dropout (Dropout)           (None, 128)               0

 dense_1 (Dense)             (None, 256)               33024

 batch_normalization_1 (Batc  (None, 256)              1024
 hNormalization)

 dropout_1 (Dropout)         (None, 256)               0

 dense_2 (Dense)             (None, 128)               32896

 batch_normalization_2 (Batc  (None, 128)              512
 hNormalization)

 dropout_2 (Dropout)         (None, 128)               0

 dense_3 (Dense)             (None, 64)                8256

 dropout_3 (Dropout)         (None, 64)                0

 dense_4 (Dense)             (None, 10)                650

=================================================================
Total params: 82,378
Trainable params: 81,354
Non-trainable params: 1,024
_________________________________________________________________

🚀 بدء التدريب...
Epoch 1/200
34/41 [=======================>......] - ETA: 0s - loss: 1.8917 - accuracy: 0.3575
Epoch 1: val_accuracy improved from -inf to 0.25688, saving model to arabic_model\best_model.h5
41/41 [==============================] - 2s 14ms/step - loss: 1.7806 - accuracy: 0.3869 - val_loss: 2.1295 - val_accuracy: 0.2569 - lr: 0.0010
Epoch 2/200
40/41 [============================>.] - ETA: 0s - loss: 0.9401 - accuracy: 0.6477
Epoch 2: val_accuracy improved from 0.25688 to 0.35780, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 6ms/step - loss: 0.9371 - accuracy: 0.6483 - val_loss: 1.8692 - val_accuracy: 0.3578 - lr: 0.0010
Epoch 3/200
39/41 [===========================>..] - ETA: 0s - loss: 0.6817 - accuracy: 0.7484
Epoch 3: val_accuracy improved from 0.35780 to 0.55046, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 7ms/step - loss: 0.6800 - accuracy: 0.7477 - val_loss: 1.5704 - val_accuracy: 0.5505 - lr: 0.0010
Epoch 4/200
39/41 [===========================>..] - ETA: 0s - loss: 0.6165 - accuracy: 0.7508
Epoch 4: val_accuracy improved from 0.55046 to 0.56575, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 7ms/step - loss: 0.6232 - accuracy: 0.7492 - val_loss: 1.2922 - val_accuracy: 0.5657 - lr: 0.0010
Epoch 5/200
32/41 [======================>.......] - ETA: 0s - loss: 0.5598 - accuracy: 0.7822
Epoch 5: val_accuracy improved from 0.56575 to 0.61774, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 7ms/step - loss: 0.5845 - accuracy: 0.7691 - val_loss: 1.0993 - val_accuracy: 0.6177 - lr: 0.0010
Epoch 6/200
32/41 [======================>.......] - ETA: 0s - loss: 0.5459 - accuracy: 0.7949
Epoch 6: val_accuracy improved from 0.61774 to 0.73089, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 6ms/step - loss: 0.5459 - accuracy: 0.7882 - val_loss: 0.8273 - val_accuracy: 0.7309 - lr: 0.0010
Epoch 7/200
35/41 [========================>.....] - ETA: 0s - loss: 0.4973 - accuracy: 0.8152
Epoch 7: val_accuracy improved from 0.73089 to 0.75535, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 7ms/step - loss: 0.5017 - accuracy: 0.8112 - val_loss: 0.6991 - val_accuracy: 0.7554 - lr: 0.0010
Epoch 8/200
34/41 [=======================>......] - ETA: 0s - loss: 0.4890 - accuracy: 0.8116
Epoch 8: val_accuracy improved from 0.75535 to 0.84098, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 8ms/step - loss: 0.5024 - accuracy: 0.8089 - val_loss: 0.5304 - val_accuracy: 0.8410 - lr: 0.0010
Epoch 9/200
29/41 [====================>.........] - ETA: 0s - loss: 0.4678 - accuracy: 0.8093
Epoch 9: val_accuracy improved from 0.84098 to 0.86239, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 6ms/step - loss: 0.4806 - accuracy: 0.8119 - val_loss: 0.4797 - val_accuracy: 0.8624 - lr: 0.0010
Epoch 10/200
39/41 [===========================>..] - ETA: 0s - loss: 0.4299 - accuracy: 0.8325
Epoch 10: val_accuracy did not improve from 0.86239
41/41 [==============================] - 0s 5ms/step - loss: 0.4245 - accuracy: 0.8341 - val_loss: 0.4433 - val_accuracy: 0.8410 - lr: 0.0010
Epoch 11/200
32/41 [======================>.......] - ETA: 0s - loss: 0.4410 - accuracy: 0.8311
Epoch 11: val_accuracy did not improve from 0.86239
41/41 [==============================] - 0s 4ms/step - loss: 0.4429 - accuracy: 0.8280 - val_loss: 0.4455 - val_accuracy: 0.8410 - lr: 0.0010
Epoch 12/200
33/41 [=======================>......] - ETA: 0s - loss: 0.4154 - accuracy: 0.8390
Epoch 12: val_accuracy did not improve from 0.86239
41/41 [==============================] - 0s 6ms/step - loss: 0.4200 - accuracy: 0.8356 - val_loss: 0.3866 - val_accuracy: 0.8624 - lr: 0.0010
Epoch 13/200
33/41 [=======================>......] - ETA: 0s - loss: 0.4238 - accuracy: 0.8475
Epoch 13: val_accuracy improved from 0.86239 to 0.87768, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 6ms/step - loss: 0.4217 - accuracy: 0.8448 - val_loss: 0.3725 - val_accuracy: 0.8777 - lr: 0.0010
Epoch 14/200
34/41 [=======================>......] - ETA: 0s - loss: 0.3619 - accuracy: 0.8667
Epoch 14: val_accuracy did not improve from 0.87768
41/41 [==============================] - 0s 6ms/step - loss: 0.3741 - accuracy: 0.8593 - val_loss: 0.4016 - val_accuracy: 0.8746 - lr: 0.0010
Epoch 15/200
31/41 [=====================>........] - ETA: 0s - loss: 0.4004 - accuracy: 0.8609
Epoch 15: val_accuracy did not improve from 0.87768
41/41 [==============================] - 0s 5ms/step - loss: 0.3991 - accuracy: 0.8547 - val_loss: 0.3410 - val_accuracy: 0.8746 - lr: 0.0010
Epoch 16/200
38/41 [==========================>...] - ETA: 0s - loss: 0.3929 - accuracy: 0.8561
Epoch 16: val_accuracy did not improve from 0.87768
41/41 [==============================] - 0s 5ms/step - loss: 0.3910 - accuracy: 0.8593 - val_loss: 0.3581 - val_accuracy: 0.8624 - lr: 0.0010
Epoch 17/200
26/41 [==================>...........] - ETA: 0s - loss: 0.3620 - accuracy: 0.8582
Epoch 17: val_accuracy did not improve from 0.87768
41/41 [==============================] - 0s 5ms/step - loss: 0.3660 - accuracy: 0.8570 - val_loss: 0.3681 - val_accuracy: 0.8654 - lr: 0.0010
Epoch 18/200
28/41 [===================>..........] - ETA: 0s - loss: 0.3218 - accuracy: 0.8761
Epoch 18: val_accuracy improved from 0.87768 to 0.92049, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 6ms/step - loss: 0.3289 - accuracy: 0.8746 - val_loss: 0.3087 - val_accuracy: 0.9205 - lr: 0.0010
Epoch 19/200
35/41 [========================>.....] - ETA: 0s - loss: 0.3511 - accuracy: 0.8607
Epoch 19: val_accuracy did not improve from 0.92049
41/41 [==============================] - 0s 6ms/step - loss: 0.3649 - accuracy: 0.8555 - val_loss: 0.3129 - val_accuracy: 0.9021 - lr: 0.0010
Epoch 20/200
32/41 [======================>.......] - ETA: 0s - loss: 0.3399 - accuracy: 0.8613
Epoch 20: val_accuracy did not improve from 0.92049
41/41 [==============================] - 0s 4ms/step - loss: 0.3431 - accuracy: 0.8654 - val_loss: 0.3147 - val_accuracy: 0.9144 - lr: 0.0010
Epoch 21/200
35/41 [========================>.....] - ETA: 0s - loss: 0.3302 - accuracy: 0.8696
Epoch 21: val_accuracy did not improve from 0.92049
41/41 [==============================] - 0s 6ms/step - loss: 0.3351 - accuracy: 0.8700 - val_loss: 0.3014 - val_accuracy: 0.8746 - lr: 0.0010
Epoch 22/200
32/41 [======================>.......] - ETA: 0s - loss: 0.3278 - accuracy: 0.8760
Epoch 22: val_accuracy did not improve from 0.92049
41/41 [==============================] - 0s 6ms/step - loss: 0.3320 - accuracy: 0.8754 - val_loss: 0.3263 - val_accuracy: 0.8807 - lr: 0.0010
Epoch 23/200
32/41 [======================>.......] - ETA: 0s - loss: 0.3324 - accuracy: 0.8770
Epoch 23: val_accuracy did not improve from 0.92049
41/41 [==============================] - 0s 6ms/step - loss: 0.3262 - accuracy: 0.8838 - val_loss: 0.2785 - val_accuracy: 0.9174 - lr: 0.0010
Epoch 24/200
31/41 [=====================>........] - ETA: 0s - loss: 0.2956 - accuracy: 0.8760
Epoch 24: val_accuracy did not improve from 0.92049
41/41 [==============================] - 0s 5ms/step - loss: 0.2911 - accuracy: 0.8800 - val_loss: 0.2757 - val_accuracy: 0.9174 - lr: 0.0010
Epoch 25/200
38/41 [==========================>...] - ETA: 0s - loss: 0.3070 - accuracy: 0.8766
Epoch 25: val_accuracy improved from 0.92049 to 0.93578, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 7ms/step - loss: 0.3015 - accuracy: 0.8784 - val_loss: 0.2750 - val_accuracy: 0.9358 - lr: 0.0010
Epoch 26/200
31/41 [=====================>........] - ETA: 0s - loss: 0.3075 - accuracy: 0.8861
Epoch 26: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 6ms/step - loss: 0.3113 - accuracy: 0.8846 - val_loss: 0.2636 - val_accuracy: 0.9266 - lr: 0.0010
Epoch 27/200
30/41 [====================>.........] - ETA: 0s - loss: 0.2998 - accuracy: 0.8917
Epoch 27: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 5ms/step - loss: 0.3219 - accuracy: 0.8777 - val_loss: 0.2527 - val_accuracy: 0.9327 - lr: 0.0010
Epoch 28/200
37/41 [==========================>...] - ETA: 0s - loss: 0.3346 - accuracy: 0.8725
Epoch 28: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 5ms/step - loss: 0.3359 - accuracy: 0.8731 - val_loss: 0.2850 - val_accuracy: 0.9266 - lr: 0.0010
Epoch 29/200
41/41 [==============================] - ETA: 0s - loss: 0.2758 - accuracy: 0.9052
Epoch 29: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 5ms/step - loss: 0.2758 - accuracy: 0.9052 - val_loss: 0.3229 - val_accuracy: 0.8746 - lr: 0.0010
Epoch 30/200
33/41 [=======================>......] - ETA: 0s - loss: 0.2703 - accuracy: 0.9062
Epoch 30: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 4ms/step - loss: 0.2829 - accuracy: 0.8998 - val_loss: 0.2555 - val_accuracy: 0.9174 - lr: 0.0010
Epoch 31/200
38/41 [==========================>...] - ETA: 0s - loss: 0.2647 - accuracy: 0.8939
Epoch 31: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 5ms/step - loss: 0.2708 - accuracy: 0.8937 - val_loss: 0.2657 - val_accuracy: 0.8991 - lr: 0.0010
Epoch 32/200
33/41 [=======================>......] - ETA: 0s - loss: 0.3253 - accuracy: 0.8750
Epoch 32: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 7ms/step - loss: 0.3327 - accuracy: 0.8784 - val_loss: 0.2691 - val_accuracy: 0.9113 - lr: 0.0010
Epoch 33/200
31/41 [=====================>........] - ETA: 0s - loss: 0.2891 - accuracy: 0.8891
Epoch 33: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 6ms/step - loss: 0.2942 - accuracy: 0.8830 - val_loss: 0.2555 - val_accuracy: 0.9113 - lr: 0.0010
Epoch 34/200
28/41 [===================>..........] - ETA: 0s - loss: 0.2716 - accuracy: 0.9018
Epoch 34: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 6ms/step - loss: 0.2652 - accuracy: 0.9052 - val_loss: 0.2272 - val_accuracy: 0.9327 - lr: 0.0010
Epoch 35/200
33/41 [=======================>......] - ETA: 0s - loss: 0.2462 - accuracy: 0.9157
Epoch 35: val_accuracy did not improve from 0.93578
41/41 [==============================] - 0s 5ms/step - loss: 0.2481 - accuracy: 0.9136 - val_loss: 0.2690 - val_accuracy: 0.9327 - lr: 0.0010
Epoch 36/200
36/41 [=========================>....] - ETA: 0s - loss: 0.2793 - accuracy: 0.9115
Epoch 36: val_accuracy improved from 0.93578 to 0.94801, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 8ms/step - loss: 0.2806 - accuracy: 0.9067 - val_loss: 0.2062 - val_accuracy: 0.9480 - lr: 0.0010
Epoch 37/200
40/41 [============================>.] - ETA: 0s - loss: 0.2743 - accuracy: 0.9008
Epoch 37: val_accuracy did not improve from 0.94801
41/41 [==============================] - 0s 6ms/step - loss: 0.2772 - accuracy: 0.8998 - val_loss: 0.2246 - val_accuracy: 0.9480 - lr: 0.0010
Epoch 38/200
38/41 [==========================>...] - ETA: 0s - loss: 0.2546 - accuracy: 0.9054
Epoch 38: val_accuracy improved from 0.94801 to 0.95107, saving model to arabic_model\best_model.h5
41/41 [==============================] - 0s 8ms/step - loss: 0.2545 - accuracy: 0.9075 - val_loss: 0.2104 - val_accuracy: 0.9511 - lr: 0.0010
Epoch 39/200
35/41 [========================>.....] - ETA: 0s - loss: 0.2410 - accuracy: 0.9080
Epoch 39: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2562 - accuracy: 0.9037 - val_loss: 0.3528 - val_accuracy: 0.8960 - lr: 0.0010
Epoch 40/200
30/41 [====================>.........] - ETA: 0s - loss: 0.2595 - accuracy: 0.9000
Epoch 40: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2740 - accuracy: 0.8976 - val_loss: 0.2644 - val_accuracy: 0.9266 - lr: 0.0010
Epoch 41/200
39/41 [===========================>..] - ETA: 0s - loss: 0.2491 - accuracy: 0.9103
Epoch 41: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 7ms/step - loss: 0.2549 - accuracy: 0.9060 - val_loss: 0.2792 - val_accuracy: 0.9144 - lr: 0.0010
Epoch 42/200
38/41 [==========================>...] - ETA: 0s - loss: 0.2532 - accuracy: 0.9178
Epoch 42: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 7ms/step - loss: 0.2496 - accuracy: 0.9167 - val_loss: 0.2484 - val_accuracy: 0.9266 - lr: 0.0010
Epoch 43/200
35/41 [========================>.....] - ETA: 0s - loss: 0.2809 - accuracy: 0.8991
Epoch 43: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2767 - accuracy: 0.9006 - val_loss: 0.2774 - val_accuracy: 0.8991 - lr: 0.0010
Epoch 44/200
37/41 [==========================>...] - ETA: 0s - loss: 0.2307 - accuracy: 0.9113
Epoch 44: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2223 - accuracy: 0.9159 - val_loss: 0.2612 - val_accuracy: 0.9174 - lr: 0.0010
Epoch 45/200
33/41 [=======================>......] - ETA: 0s - loss: 0.2234 - accuracy: 0.9167
Epoch 45: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2204 - accuracy: 0.9174 - val_loss: 0.1984 - val_accuracy: 0.9450 - lr: 0.0010
Epoch 46/200
27/41 [==================>...........] - ETA: 0s - loss: 0.2366 - accuracy: 0.9201
Epoch 46: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2241 - accuracy: 0.9228 - val_loss: 0.2549 - val_accuracy: 0.9052 - lr: 0.0010
Epoch 47/200
32/41 [======================>.......] - ETA: 0s - loss: 0.2216 - accuracy: 0.9238
Epoch 47: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2086 - accuracy: 0.9266 - val_loss: 0.2339 - val_accuracy: 0.9327 - lr: 0.0010
Epoch 48/200
31/41 [=====================>........] - ETA: 0s - loss: 0.2171 - accuracy: 0.9254
Epoch 48: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2035 - accuracy: 0.9258 - val_loss: 0.1707 - val_accuracy: 0.9358 - lr: 0.0010
Epoch 49/200
39/41 [===========================>..] - ETA: 0s - loss: 0.2229 - accuracy: 0.9215
Epoch 49: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2230 - accuracy: 0.9235 - val_loss: 0.2239 - val_accuracy: 0.9388 - lr: 0.0010
Epoch 50/200
36/41 [=========================>....] - ETA: 0s - loss: 0.2146 - accuracy: 0.9123
Epoch 50: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2128 - accuracy: 0.9128 - val_loss: 0.2365 - val_accuracy: 0.9358 - lr: 0.0010
Epoch 51/200
30/41 [====================>.........] - ETA: 0s - loss: 0.2040 - accuracy: 0.9323
Epoch 51: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 4ms/step - loss: 0.2147 - accuracy: 0.9297 - val_loss: 0.2046 - val_accuracy: 0.9419 - lr: 0.0010
Epoch 52/200
29/41 [====================>.........] - ETA: 0s - loss: 0.2370 - accuracy: 0.9106
Epoch 52: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2307 - accuracy: 0.9128 - val_loss: 0.1691 - val_accuracy: 0.9419 - lr: 0.0010
Epoch 53/200
41/41 [==============================] - ETA: 0s - loss: 0.2329 - accuracy: 0.9197
Epoch 53: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 6ms/step - loss: 0.2329 - accuracy: 0.9197 - val_loss: 0.1865 - val_accuracy: 0.9511 - lr: 0.0010
Epoch 54/200
39/41 [===========================>..] - ETA: 0s - loss: 0.2462 - accuracy: 0.9127
Epoch 54: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 7ms/step - loss: 0.2454 - accuracy: 0.9144 - val_loss: 0.2535 - val_accuracy: 0.9205 - lr: 0.0010
Epoch 55/200
37/41 [==========================>...] - ETA: 0s - loss: 0.2263 - accuracy: 0.9215
Epoch 55: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2279 - accuracy: 0.9220 - val_loss: 0.2433 - val_accuracy: 0.9358 - lr: 0.0010
Epoch 56/200
39/41 [===========================>..] - ETA: 0s - loss: 0.2032 - accuracy: 0.9239
Epoch 56: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.2100 - accuracy: 0.9220 - val_loss: 0.2361 - val_accuracy: 0.9297 - lr: 0.0010
Epoch 57/200
39/41 [===========================>..] - ETA: 0s - loss: 0.2035 - accuracy: 0.9263
Epoch 57: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 5ms/step - loss: 0.1986 - accuracy: 0.9289 - val_loss: 0.2478 - val_accuracy: 0.9358 - lr: 0.0010
Epoch 58/200
33/41 [=======================>......] - ETA: 0s - loss: 0.2088 - accuracy: 0.9176Restoring model weights from the end of the best epoch: 38.

Epoch 58: val_accuracy did not improve from 0.95107
41/41 [==============================] - 0s 7ms/step - loss: 0.2266 - accuracy: 0.9136 - val_loss: 0.2250 - val_accuracy: 0.9419 - lr: 0.0010
Epoch 58: early stopping

📊 تقييم الموديل على بيانات الاختبار:
  ✅ Accuracy: 91.20%
  📉 Loss: 0.2574
13/13 [==============================] - 0s 2ms/step

📋 تقرير التصنيف:
              precision    recall  f1-score   support

           أ       0.95      1.00      0.98        40
           ب       0.87      0.98      0.92        42
           ت       0.98      1.00      0.99        40
           ث       1.00      0.98      0.99        43
           ج       1.00      0.85      0.92        41
           ح       0.85      1.00      0.92        41
           خ       1.00      0.97      0.99        40
           د       0.95      0.53      0.68        40
           ذ       0.95      0.95      0.95        42
           ر       0.67      0.85      0.75        40

    accuracy                           0.91       409
   macro avg       0.92      0.91      0.91       409
weighted avg       0.92      0.91      0.91       409

Traceback (most recent call last):
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\train_model.py", line 182, in <module>
    plt.show()
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\matplotlib\pyplot.py", line 613, in show
    return _get_backend_mod().show(*args, **kwargs)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\matplotlib\backend_bases.py", line 3550, in show
    cls.mainloop()
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\matplotlib\backends\_backend_tk.py", line 572, in start_main_loop
    first_manager.window.mainloop()
  File "C:\Users\`mohamedonly1\AppData\Local\Programs\Python\Python310\lib\tkinter\__init__.py", line 1458, in mainloop
    self.tk.mainloop(n)
KeyboardInterrupt
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-26 23:26:09.371876: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:26:09.372270: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Traceback (most recent call last):
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 47, in <module>
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\tensorflow\lite\python\interpreter.py", line 455, in __init__
    _interpreter_wrapper.CreateWrapperFromFile(
ValueError: Could not open 'arabic_model/arabic_sign_model.tflite'.
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> dir arabic_model


    Directory: C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\arabic_model


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2/26/2026  11:23 PM        1057792 best_model.h5
-a----         2/26/2026  11:23 PM          73236 confusion_matrix.png


(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> dir arabic_data


    Directory: C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\arabic_data


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2/26/2026  11:20 PM        1563851 arabic_keypoints.csv
-a----         2/26/2026  11:02 PM            195 arabic_labels.csv


(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python -c "
>> import tensorflow as tf
>> model = tf.keras.models.load_model('arabic_model/best_model.h5')
>> converter = tf.lite.TFLiteConverter.from_keras_model(model)
>> tflite_model = converter.convert()
>> open('arabic_model/arabic_sign_model.tflite', 'wb').write(tflite_model)
>> print('Done!')
>> "
2026-02-26 23:27:28.794851: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:27:28.795091: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2026-02-26 23:27:35.077642: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:27:35.084435: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
2026-02-26 23:27:35.092440: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublasLt64_11.dll'; dlerror: cublasLt64_11.dll not found
2026-02-26 23:27:35.096635: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2026-02-26 23:27:35.103714: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2026-02-26 23:27:35.110590: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusolver64_11.dll'; dlerror: cusolver64_11.dll not found
2026-02-26 23:27:35.116408: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusparse64_11.dll'; dlerror: cusparse64_11.dll not found
2026-02-26 23:27:35.121437: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
2026-02-26 23:27:35.121561: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2026-02-26 23:27:35.124856: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-02-26 23:27:39.194587: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:362] Ignored output_format.
2026-02-26 23:27:39.194839: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:365] Ignored drop_control_dependency.
2026-02-26 23:27:39.196738: I tensorflow/cc/saved_model/reader.cc:45] Reading SavedModel from: C:\Users\`MOHAM~1\AppData\Local\Temp\tmprhbm_gwc
2026-02-26 23:27:39.202088: I tensorflow/cc/saved_model/reader.cc:89] Reading meta graph with tags { serve }
2026-02-26 23:27:39.202176: I tensorflow/cc/saved_model/reader.cc:130] Reading SavedModel debug info (if present) from: C:\Users\`MOHAM~1\AppData\Local\Temp\tmprhbm_gwc
2026-02-26 23:27:39.217451: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
2026-02-26 23:27:39.219513: I tensorflow/cc/saved_model/loader.cc:229] Restoring SavedModel bundle.
2026-02-26 23:27:39.330893: I tensorflow/cc/saved_model/loader.cc:213] Running initialization op on SavedModel bundle at path: C:\Users\`MOHAM~1\AppData\Local\Temp\tmprhbm_gwc
2026-02-26 23:27:39.351907: I tensorflow/cc/saved_model/loader.cc:305] SavedModel load for tags { serve }; Status: success: OK. Took 155153 microseconds.
2026-02-26 23:27:39.429538: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
Done!
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-26 23:27:45.217972: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:27:45.218182: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✅ التطبيق جاهز!
  SPACE: إضافة مسافة | BACKSPACE: حذف | C: مسح الكلمة | ESC: خروج
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Traceback (most recent call last):
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 240, in <module>
    main()
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 135, in main
    results = hands.process(rgb_image)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\mediapipe\python\solutions\hands.py", line 153, in process
    return super().process(input_data={'image': image})
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\mediapipe\python\solution_base.py", line 335, in process
    self._graph.wait_until_idle()
KeyboardInterrupt
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-26 23:29:46.428881: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:29:46.429012: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✅ التطبيق جاهز!
  SPACE: إضافة مسافة | BACKSPACE: حذف | C: مسح الكلمة | ESC: خروج
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Traceback (most recent call last):
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 240, in <module>
    main()
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 135, in main
    results = hands.process(rgb_image)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\mediapipe\python\solutions\hands.py", line 153, in process
    return super().process(input_data={'image': image})
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\mediapipe\python\solution_base.py", line 335, in process
    self._graph.wait_until_idle()
KeyboardInterrupt
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-26 23:31:03.536375: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:31:03.536564: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✅ التطبيق جاهز!
  SPACE: إضافة مسافة | BACKSPACE: حذف | C: مسح الكلمة | ESC: خروج
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Traceback (most recent call last):
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 240, in <module>
    if current_word:
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 206, in main
    (10, 8), font_size=40, color=conf_color)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 89, in put_arabic_text
    font = ImageFont.truetype(FONT_PATH, font_size)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\PIL\Image.py", line 3335, in fromarray
    return frombuffer(mode, size, obj, "raw", rawmode, 0, 1)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\PIL\Image.py", line 3226, in frombuffer
    return frombytes(mode, size, data, decoder_name, args)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\PIL\Image.py", line 3153, in frombytes
    im = new(mode, size)
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\venv310\lib\site-packages\PIL\Image.py", line 3118, in new
    return im._new(core.fill(mode, size, color))
KeyboardInterrupt
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-26 23:35:04.083373: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:35:04.083627: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✅ التطبيق جاهز!
  SPACE: مسافة | BACKSPACE: حذف | C: مسح | ESC: خروج
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Traceback (most recent call last):
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 256, in <module>
    main()
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 224, in main
    debug_image = put_arabic_text(debug_image, word_text,
  File "C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main\app_arabic.py", line 87, in put_arabic_text
    img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
KeyboardInterrupt
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-26 23:40:33.048680: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-26 23:40:33.049726: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✅ التطبيق جاهز!
  SPACE: مسافة | BACKSPACE: حذف | C: مسح | ESC: خروج
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

📝 آخر كلمة: بترأجخثثأأخأ
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main> python app_arabic.py
2026-02-27 00:05:37.296740: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2026-02-27 00:05:37.297161: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✅ التطبيق جاهز!
  SPACE: مسافة | BACKSPACE: حذف | C: مسح | ESC: خروج
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

📝 آخر كلمة: أبتجرذخحب
(venv310) PS C:\Users\`mohamedonly1\Desktop\hand-gesture-recognition-mediapipe-main>