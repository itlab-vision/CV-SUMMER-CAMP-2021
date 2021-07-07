import argparse
import cv2
import sys

def make_cat_passport_image(input_image_path, input_image2_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    #print(image.shape)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray.shape, gray.dtype)
    #cv2.imshow('test1', gray)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)
    #cv2.imshow('test2', gray)

    # Resize image

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    print(rects)

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    cv2.imshow("result_frame", image)

    # Crop image
    x, y, w, h = rects[0]
    ROIimage = image[y:y+h, x:x+w]
    
    # Save result image to file

    #image3 = image[0 : 300 : 2, : :2]
    #cv2.imshow("test5", image3)
    #image3[0 : 300 : 2, : :2] = (0,0,255)
    #cv2.imshow("test6", image3)

    cv2.imshow("ROIimage", ROIimage)
    cv2.imwrite('out.jpg', ROIimage)

    # Cat passport
    passport = cv2.imread(input_image2_path)

    #Text-on-Passport
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(passport,'TestCat',(90,218), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'Cat',(90,231), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'Unknown',(90,245), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'Male',(90,260), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'02/01/2019',(110,272), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'White+Black',(90,285), font, 0.4,(0,255,0),1,cv2.LINE_AA)

    cv2.putText(passport,'EA010231',(300,91), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'05/03/2019',(300,130), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'Unknown',(300,170), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'001232',(300,212), font, 0.4,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(passport,'13/03/2019',(300,251), font, 0.4,(0,255,0),1,cv2.LINE_AA)

    # Cat-to-Passport
    height,width,depth = ROIimage.shape
    CorCoeff = 47

    for h1 in range(height):
        for w1 in range(width):
            for c1 in range(depth):
                passport[h1 + CorCoeff ,w1 + CorCoeff ,c1] = ROIimage[h1,w1,c1]    

    # Save result image to file
    cv2.imshow("Passport_out", passport) 
    cv2.imwrite('passport_out.jpg', passport)
 

    # WaitKey
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    args.add_argument('-i2', '--input2', type=str, required=True,
                      help='Required. Path to input image2')
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.input2, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
