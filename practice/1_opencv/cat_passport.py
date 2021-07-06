import sys
import argparse
import cv2


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)

    # Resize image

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    print(rects)

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Display result image
    cv2.imshow("Find a cat", image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Save result image to file
    cv2.imwrite('out.jpg', image)

    #----------------------------------Filling out the passport---------------------------------+
    input_image_path = 'pet_passport.png'
    passport = cv2.imread(input_image_path)

    rows, cols, chanels = image.shape
    image = cv2.addWeighted(passport[45:45+rows, 48:48+cols],0,image,0.7,0)
    passport[45:45+rows,48:48+cols] = image

    cv2.putText(passport, "Nika", (88,215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(passport, "Cat", (88,230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(passport, "Karat", (88,245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(passport, "Male", (88,260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(passport, "01.04.2019", (108,272), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(passport, "Short", (93,285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite('pet_passport_photo.jpg', passport)
    
    cv2.imshow("Passport", passport)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    #-------------------------------------------------------------------------------------------+
    print('"The program works correctly"')
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

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
