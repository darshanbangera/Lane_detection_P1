import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# printing out some stats and plotting
print('This image is:', type(image), 'with dimentions:', image.shape)
xsize = image.shape[1]
ysize = image.shape[0]


def grayscale(img):
    '''
    :param img: Image in 3 color channels
    :return: Applies the grayscale transform ande returns image with only one color channel
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """
    :param img: Grayscale image
    :param low_threshold: pixels below this treshold will be rejected
    :param high_threshold: pixels above this treshold will be treated as strong edges
    :return: Applies canny transform on the image
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img,vertices):
    """
    :param img: image
    :param vertices: Vertices of polygon
    :return: Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are non zero
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image


def draw_lines(img, lines, color=[0,0,255], thickness=5):
    x_left,y_left,x_right,y_right = ([] for i in range(4))
    for line in lines:
        for x1,y1,x2,y2 in line:
            # selecting the right side lines
            if x1<xsize/2:
                x_left.extend([x1,x2])
                y_left.extend([y1,y2])
            # selecting the left side lines
            elif x1>xsize/2:
                x_right.extend([x1, x2])
                y_right.extend([y1, y2])
    # numpy array of x and y array of left side points
    x_left = np.asarray(x_left)
    y_left = np.asarray(y_left)
    # using polyfit function to obtain the best fitting line for the set of points
    m_left,b_left = np.polyfit(x_left, y_left, 1)

    x_right = np.asarray(x_right)
    y_right = np.asarray(y_right)
    m_right, b_right = np.polyfit(x_right, y_right, 1)

    x1 = int((320 - b_right) / m_right)
    x2 = int((ysize - b_right) / m_right)
    x3 = int((320 - b_left) / m_left)
    x4 = int((ysize - b_left) / m_left)

    cv2.line(img, (x1, 320), (x2, ysize), color, thickness)
    cv2.line(img, (x3, 320), (x4, ysize), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    :param img: canny transformed image
    :param rho: distance in hough space
    :param theta: angular resolution in hough space
    :param threshold: minimum number of votes (intersections in a given grid cell) a candidate line needs to have to make it into the output.
    :param min_line_len: minimum length of a line (in pixels) the code will accept in the output
    :param max_line_gap: minimum distance between segments (in pixels) the code will accept in the output
    :returns: An image with hough lines drawn
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0],img.shape[1], 3), dtype= np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def lane_finder(input_video, output_video):
    video = cv2.VideoCapture(input_video)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*'MJPG'),fps, size)
    while(video.isOpened()):
        ret, frame = video.read()
        threshold = 15
        vertices = np.asarray([[100, ysize], [900, ysize], [520, 323], [440, 323]], np.int32)
        rho = 2
        theta = np.pi / 180
        min_line_length = 40
        max_line_gap = 20
        gray = grayscale(frame)
        can = canny(gray, 60, 180)
        can = region_of_interest(can, [vertices])
        hough = hough_lines(can, rho, theta, threshold, min_line_length, max_line_gap)
        res = weighted_img(hough, frame)
        result.write(res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


input_white = "test_videos/solidWhiteRight.mp4"
input_yellow = "test_videos/solidYellowLeft.mp4"
output_white = 'test_videos_output/solidWhiteRight.avi'
output_yellow = 'test_videos_output/solidYellowLeft.avi'

lane_finder(input_white,output_white)
lane_finder(input_yellow,output_yellow)

