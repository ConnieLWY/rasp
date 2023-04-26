import easyocr
from roboflow import Roboflow
import cv2


text_reader = easyocr.Reader(['en']) #Initialzing the ocr

rf = Roboflow(api_key="FGBImpoSUaJ2OVdvZjVI")
project = rf.workspace().project("mplates")
model = project.version(2).model

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    pos = model.predict(frame, confidence=50, overlap=50).json()
    if pos['predictions']:
        if pos['predictions'][0] is not None:
            x_min = int(pos['predictions'][0]['x'] - pos['predictions'][0]['width'] / 2)
            y_min = int(pos['predictions'][0]['y'] - pos['predictions'][0]['height'] / 2)
            x_max = int(pos['predictions'][0]['x'] + pos['predictions'][0]['width'] / 2)
            y_max = int(pos['predictions'][0]['y'] + pos['predictions'][0]['height'] / 2)
            # Crop the car from the frame using the bounding box coordinates
            lp = frame[y_min:y_max, x_min:x_max]
            gray = cv2.cvtColor(lp, cv2.COLOR_BGR2RGB)

            results = text_reader.readtext(gray)
            all_text = ""

            for (bbox, text, prob) in results:
                all_text += text
            print(all_text)
         
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


