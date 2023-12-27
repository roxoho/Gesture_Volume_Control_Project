import time
import cv2
import mediapipe as mp


class hand_Detector():
    def __init__(self,mode=False, maxHands = 2, detectionCon = 0.5, trackingCon = 0.5, modelComplex = 1 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.modelComplex = modelComplex
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handno = 0, draw = True):
        lmlist = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handno]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmlist


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = hand_Detector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm = detector.findPosition(img, draw=False)
        if len(lm) != 0:
            print(lm[0])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
