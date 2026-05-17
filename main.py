import numpy as np
import cv2
import matplotlib.pyplot as plt
from Lab17_NMS_IoU import NMS,IoU
from Read_Annotation import load_data
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from  keras.optimizers import Adam
from keras.models import load_model,Sequential

input_shape = (128, 128, 3)
model = Sequential()
model.add(tf.keras.Input(shape=input_shape))         
model.add(Conv2D(32,  (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,  (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))                               
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

x, t = load_data()
x = x.astype('float32') / 255.0
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=42)

op = Adam(learning_rate=0.001)                       
model.compile(optimizer=op, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20,     
                    validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")
model.save("Base_model_keras.keras")                  

#########################################################################################

def RCNN(image, model):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    results = ss.process()

    results = results[:500]
    print(f"Processing {len(results)} ROIs...")

    rois, boxes = [], []

    for box in results:
        x1, y1 = box[0], box[1]
        x2, y2 = box[0] + box[2], box[1] + box[3]
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi = cv2.resize(roi, (128, 128))
        rois.append(roi)
        boxes.append([x1, y1, x2, y2])

    # Batch predict — the key fix
    rois_array = np.array(rois, dtype='float32') / 255.0
    predictions = model.predict(rois_array, batch_size=64, verbose=0)
    print("Prediction done.")

    positive_boxes = [boxes[i] for i, pred in enumerate(predictions) if pred[0] > 0.7]

    if not positive_boxes:
        print("No number plates detected.")
        return

    cleaned = NMS(np.array(positive_boxes), 0.5)
    cleaned = np.asarray(cleaned, dtype='int')

    output = image.copy()
    for box in cleaned:
        cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(output, "Plate", (box[0], box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected: {len(cleaned)} plate(s)")
    plt.axis('off')
    plt.show()
##############################################################################################

test_img=cv2.imread('numberplates/Cars0.png')
if test_img is not None:
    test_img=cv2.resize(test_img,(400,300))
    rcnn_model=tf.keras.models.load_model('Base_model_keras.keras')
    RCNN(test_img,rcnn_model)
else:
    print("Test image 'numberplates/Cars0.png' not found!")

