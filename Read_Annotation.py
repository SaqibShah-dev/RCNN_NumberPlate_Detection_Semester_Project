import cv2
import pandas as pd
import numpy as np
import os
from Lab17_NMS_IoU import NMS, IoU

# 1. Load the annotations
annotations = pd.read_csv('numberplates/annotations.csv')

# 2. Extract filenames and bounding boxes
allnames = annotations.iloc[:, 0].values
box_list = annotations.iloc[:, [3, 4, 5, 6]].values


def load_data():
    annotations = pd.read_csv('numberplates/annotations.csv')
    x = []
    y = []
    image_folder = 'numberplates/'

    for index, row in annotations.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Positive sample (Ground Truth Plate)
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        roi = image[y1:y2, x1:x2]
        if roi.size == 0: continue
        roi = cv2.resize(roi, (128, 128))
        x.append(roi)
        y.append(1)

        # Negative sample using Hard Negative Mining via Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        results = ss.process()
        
        gt_box = [x1, y1, x2, y2]
        neg_count = 0
        for box in results:
            nx1, ny1 = box[0], box[1]
            nx2, ny2 = box[0] + box[2], box[1] + box[3]
            
            # Avoid tiny noise boxes
            if box[2] < 20 or box[3] < 20: continue
            
            iou = IoU([nx1, ny1, nx2, ny2], gt_box)
            if iou < 0.1:
                neg_roi = image[ny1:ny2, nx1:nx2]
                if neg_roi.size > 0:
                    neg_roi = cv2.resize(neg_roi, (128, 128))
                    x.append(neg_roi)
                    y.append(0)
                    neg_count += 1
            if neg_count >= 5: # Get up to 5 negative samples per positive sample
                break

    return np.array(x), np.array(y)


if __name__ == "__main__":
    # 3. Identify unique images to process them one by one
    allnames = annotations.iloc[:, 0].values
    box_list = annotations.iloc[:, [3, 4, 5, 6]].values
    unique_images = np.unique(allnames)

    # Path to where your images are stored
    image_folder = 'numberplates/'

    for img_name in unique_images:
        # Find indices of all boxes belonging to this specific image
        indices = np.where(allnames == img_name)[0]
        boxes_for_image = box_list[indices]

        # --- Apply Non-Maximum Suppression ---
        pick_boxes = NMS(boxes_for_image, overlapThreshold=0.3)

        # Load image for visualization
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not load {img_name}")
            continue

        # 4. Draw the results
        for (x1, y1, x2, y2) in boxes_for_image:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        for (x1, y1, x2, y2) in pick_boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, "Filtered Plate", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Display the output
        cv2.imshow('NMS Result: Red=Original, Green=Filtered', image)

        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
