import numpy as np 
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)
 
emotion_dict = {0: "Angry",
                1: "Disgusted",
                2: "Fearful",
                3: "Happy",
                4: "Neutral",
                5: "Sad",
                6: "Surprised"
                }

cap = cv2.VideoCapture(0)
while(True):
	# Find haar cascade to draw bounding box around face
	ret, frame = cap.read()
	if not ret:
		break
	facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)
		roi_gray = gray[y:y+h,x:x+w]
		cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray,(48,48)),-1),0)
		prediction = model.predict(cropped_img)
		maxindex = int(np.argmax(prediction))
		cv2.putText(frame, emotion_dict[maxindex],(x+20,y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation=cv2.INTER_CUBIC))
	if(cv2.waitKey(1) & (0xFF == ord('q'))):
		break


cap.release()
cv2.destroyAllWindows()

