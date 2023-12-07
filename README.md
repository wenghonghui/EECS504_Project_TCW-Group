## License plate location and recognition

Here is the Umich EECS504 Project By TCW Group. You could get more information through this [PDF](Report.pdf)

## USAGE

This is a two stage lightweight and robust license plate recognition in MTCNN and LPRNet using Pytorch. 

The modified MTCNN structure is presented as below. Only proposal net (Pnet) and output net (Onet) are used in this work since it is found that skipping Rnet will not hurt the accuracy in this case.  The Onet accepts 24(height) x 94(width) BGR image which is consistent with input for LPRNet. 

###  Training on MTCNN
* Download the [CCPD](https://github.com/detectRecog/CCPD) data and put it into 'ccpd' folder
* run 'MTCNN/data_set/preprocess.py' to split training data and validation data and put in "ccpd_train" and "ccpd_val" folders respectively.
* run 'MTCNN/data_preprocessing/gen_Pnet_train_data.py', 'MTCNN/data_preprocessing/gen_Onet_train_data.py','MTCNN/data_preprocessing/assemble_Pnet_imglist.py', 'MTCNN/data_preprocessing/assemble_Onet_imglist.py' for training data preparation.
* run 'MTCNN/train/Train_Pnet.py' and 'MTCNN/train/Train_Onet.py

### Training on LPRNet
* run 'LPRNet/data/preprocess.py' to prepare the dataset
* run 'LPRNet/LPRNet_Train.py' for training 

### Test
* run 'MTCNN/MTCNN.py' for license plate detection
* run 'LPRNet/LPRNet_Test.py' for license plate recognition
* run 'main.py' for both
