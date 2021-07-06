import sys
import argparse
import cv2


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    #print(image.shape)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray.shape, gray.dtype)
    
    # Normalize image intensity
    gray = cv2.equalizeHist(gray)
    #print(gray.shape, gray.dtype)
    
    # Resize image
    resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_AREA)
    #print(resized.shape)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    #print(rects)

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    #cv2.imshow("window_name", image)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]
    #cv2.imshow("window_name2", image)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    
    # Save result image to file
    cv2.imwrite('out.jpg', image)
    
    # Additional_task
    image2 = cv2.imread('pet_passport.png')
    print(image.shape)
    image2[50:(50+140),50:(50+140)] = image
    
    #Name
    cv2.putText(image2, 'Tom', (127, 213), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

    #Species
    cv2.putText(image2, 'Siamese cat', (110, 247), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    
    #Date_of_Birthday
    cv2.putText(image2, '22.02.2018', (116, 274), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    
    #Tattoo number
    cv2.putText(image2, '1234567890', (277, 210), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    
    #Date of Microchipping
    cv2.putText(image2, '21.02.2019', (280, 130), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    
    #Sex
    cv2.putText(image2, 'Female', (120, 260), cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    
    
    cv2.imshow("window_name", image2)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

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
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)
    image = cv2.imread('pet_passport.png')

    return 0


if __name__ == '__main__':
     sys.exit(main() or 0)
