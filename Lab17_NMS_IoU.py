
import numpy as np

def NMS(boxes, overlapThreshold):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    pick = []

    x1 = boxes[:, 0]  # top left x pixel of all boxes
    y1 = boxes[:, 1]  # top left y pixel of all boxes
    x2 = boxes[:, 2]  # bottom right x pixel of all boxes
    y2 = boxes[:, 3]  # bottom right y pixel of all boxes
    area = (x2 - x1) * (y2 - y1) # width * height

    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThreshold)[0])))

    return boxes[pick]

###########################################################################

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0]) # top left x pixel
    yA = max(boxA[1], boxB[1]) # top left y pixel
    xB = min(boxA[2], boxB[2]) # bottom right x pixel
    yB = min(boxA[3], boxB[3]) # bottom right y pixel

    interArea = max(0, xB - xA) * max(0, yB - yA) # width * height


    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


