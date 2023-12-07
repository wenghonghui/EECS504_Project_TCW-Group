import sys
sys.path.append('./LPRNet')
sys.path.append('./MTCNN')
from LPRNet_Test import *
from MTCNN import *
import numpy as np
import argparse
import torch
import time
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN & LPR Demo')
    parser.add_argument("-image", help='image path', default='test/test_7.png', type=str)
    parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum face to be detected", default=(50, 15), type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('LPRNet/weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    lprnet.eval()
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('LPRNet/weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    STN.eval()
    
    print("Successful to build LPR network!")
    
    since = time.time()
    image = cv2.imread(args.image)
    image = cv2.resize(image, (0, 0), fx = args.scale, fy = args.scale, interpolation=cv2.INTER_CUBIC)
    bboxes = create_mtcnn_net(image, args.mini_lp, device, p_model_path='MTCNN/weights/pnet_Weights', o_model_path='MTCNN/weights/onet_Weights')
    
    # judge if bboxes is empty
    if bboxes is None:
        print("Can't detect any face!")
        exit(0)
    print(bboxes)

    # judge if bboxes has negative value
    if int(bboxes[0, 0]) < 0:
        bboxes[0, 0] = 0
    if int(bboxes[0, 1]) < 0:
        bboxes[0, 1] = 0

    cropped_image = image[int(bboxes[0, 1]):int(bboxes[0, 3]), int(bboxes[0, 0]):int(bboxes[0, 2])]
    cv2.imshow('cropped_image', cropped_image)
    cv2.imwrite('test/test_cropped_7.png', cropped_image)
    
    for i in range(bboxes.shape[0]):
         
        bbox = bboxes[i, :4]
        x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
        w = int(x2 - x1 + 1.0)
        h = int(y2 - y1 + 1.0)
        img_box = np.zeros((h, w, 3))
        img_box = image[y1:y2+1, x1:x2+1, :]
        im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
        transfer = STN(data)
        preds = lprnet(transfer)
        preds = preds.cpu().detach().numpy()  # (1, 68, 18)    
        labels, pred_labels = decode(preds, CHARS)
    
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # image = cv2ImgAddText(image, labels[0], (x1, y1-12), textColor=(0, 0, 0), textSize=15)
        image = cv2ImgAddText(image, labels[0], bbox)
    
    print("model inference in {:2.3f} seconds".format(time.time() - since))      
    image = cv2.resize(image, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', image)
    cv2.imwrite('test/test_result_7.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()