import argparse
import cv2
import sys
from PIL import Image


def make_cat_passport_image(input_image_path, haar_model_path, input_passport_name):
    # Read image

    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    # print(image.shape)

    # Convert image to gropenvino-virtual-environments\bin\activate.batayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape,gray.dtype)

    # Normalize image intensity

    gray = cv2.equalizeHist(gray)
    # Resize image

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(75, 75))
    # print(rects)
    # Draw bounding box

    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + 170, y + 138), (255, 255, 255), 4)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    # image = image[0:300:2, ::2] = (0,0,255)
    # cv2.imshow("window name", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # Crop image

    x, y, w, h = rects[0]
    image = image[y:y + 138, x:x + 170]

    # Save result image to file


    cv2.imwrite('out.jpg', image)

    create_passport(image,input_passport_name)
    return


def create_passport(image,input_passport_name):
    pasport = cv2.imread(input_passport_name)

    rows, cols, channels = image.shape

    image = cv2.addWeighted(pasport[45:45 + rows, 30:30 + cols], 0, image, 0.5, 0)

    pasport[45:45 + rows, 30:30 + cols] = image

    cv2.putText(pasport, "Jesse", (80, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("hi", pasport)
    cv2.waitKey()
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
    args.add_argument('-p','--passport', type= str,required=True, help = "Required. Path to input image passport")
    return parser


def main():
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model,args.passport)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
