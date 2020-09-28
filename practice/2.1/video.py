import cv2

def main():

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        cv2.imshow('Video feed', img)
        
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()