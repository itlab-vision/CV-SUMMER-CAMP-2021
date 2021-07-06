import sys
import argparse
import cv2


def make_cat_passport_image(input_image_path, haar_model_path, passport_path):

    # Read image
    image = cv2.imread(input_image_path)
    # print(image.shape)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray.shape, gray.dtype)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)

    # Resize image
    # resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_AREA)


    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    # print(rects)

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # image[0:300:2,::2] = (0, 0, 255)

    # Display result image
    # cv2.imshow("window_name", image)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Save result image to file
    cv2.imwrite('out.jpg', image)




    # show passport with cat
    passport = cv2.imread(passport_path)
    w,h = 170,139
    image = cv2.resize(image,(w,h), interpolation = cv2.INTER_AREA)
    for x in range(h):
        for y in range(w):
            passport[x+48,y+31] = image[x,y]

    cv2.putText(passport, 'OpenCV', (86,217), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (0,0,0), 0, cv2.LINE_AA)

    cv2.putText(passport, '19.01.2038', (110,272), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (0,0,0), 0, cv2.LINE_AA)
    cv2.imshow("window_name", passport)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    print('"Finished"!!!')
    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-p', '--passport', type=str, required=True,
                      help='Required. Path to passport image.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model, args.passport)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
