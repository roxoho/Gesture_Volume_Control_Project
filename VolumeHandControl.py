import cv2
import time
import numpy
import numpy as np

import handTrackingModule as HTM
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = HTM.hand_Detector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-5.0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:

    success, img = cap.read()
    img = detector.findHands(img,draw=False)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x2+x1)//2, (y2+y1)//2

        cv2.circle(img,(x1, y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img,(x2, y2), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img,(cx, cy), 15, (255,0,255), cv2.FILLED)
        cv2.line(img,(x1, y1),(x2, y2),(255,0,255),3)

        length = math.hypot(x2-x1,y2-y1)

        if length<50:
             cv2.circle(img,(cx, cy), 15, (0,255,0), cv2.FILLED)

        #hand range = 25 - 350
        #vol range = -65 - 0
        vol = np.interp(length,[30,220],[minVol,maxVol])
        volBar = np.interp(length,[30,220],[400,150])
        volPer = np.interp(length,[30,220],[0,100])
        #print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img,(50,150),(85,400),(255, 0, 0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255, 0, 0),cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
