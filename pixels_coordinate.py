import cv2


# function to display the pixel coordinates
# of the points clicked on the image
ix,iy = -1,-1
def click_event(event, x, y, flags, params):
    global ix, iy
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        ix, iy = x, y
        print('dentro da funcao', ix, iy)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', frame)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        cv2.putText(frame, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', frame)


# driver function
if __name__ == "__main__":
    # reading the image
    frame = cv2.imread('100_0860_0026.JPG')
    height, width = frame.shape[:2]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', width, height)
    #cv2.imshow('image', frame)
    # img = cv2.imread('100_0860_0026.JPG', 1)

    # displaying the image
    # cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    print('fora da funcao', ix, iy)


    cv2.imshow('image', frame)
    k = cv2.waitKey(5000) & 0xFF
    print('fora da funcao2', ix, iy)

    # wait for a key to be pressed to exit
    #cv2.waitKey(0)

    # close the window
    #cv2.destroyAllWindows()