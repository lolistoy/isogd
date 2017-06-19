import numpy as np

def mask_unset(rects, img):
    H, W = img.shape[:2]

    mask = np.ones((H, W), bool)
    for rect in rects:
        l = int(max(min(rect[0], W-1), 0))
        u = int(max(min(rect[1], H-1), 0))
        r = int(max(min(rect[2], W-1), 0))
        b = int(max(min(rect[3], H-1), 0))
        mask[u:b, l:r] = False
    img = img.copy()
    if np.sum(mask) < H * W * 0.95:
        img[mask, :] = 0
    return img

def get_hand_rects(img_list, pose_list):
    lhrect_list = []
    rhrect_list = []
    shoulder_w = np.mean(abs(pose_list[:, 2, 0] - pose_list[:, 5, 0]))

    for ind in range(len(img_list)):
        # face location
        x_list = pose_list[ind, 14:16, 0]
        y_list = pose_list[ind, 14:16, 1]
        x_list = x_list[abs(x_list) > 10]
        y_list = y_list[abs(y_list) > 10]
        if len(x_list) > 0 and len(y_list) > 0:
            for hand_str in ['right', 'left']:
                if hand_str == 'right':
                    elbow_ind = 3
                    hand_ind = 4
                else:
                    elbow_ind = 6
                    hand_ind = 7

                elbow = pose_list[ind, elbow_ind, :2]
                hand = pose_list[ind, hand_ind, :2]
                if hand[0] == 0 and hand[1] == 0:
                    hand_rect = (0, 0, 0, 0)
                else:
                    v = hand - elbow
                    v = np.sign(v)
                    v = v / (np.linalg.norm(v) + 1e-3)
                    hand1 = hand + shoulder_w * 0.25 * v - shoulder_w * 0.80 * np.array([1, 1])
                    hand2 = hand + shoulder_w * 0.25 * v + shoulder_w * 0.80 * np.array([1, 1])
                    hand_rect = (hand1[0], hand1[1], hand2[0], hand2[1])

                if hand_str == 'right':
                    rhrect_list.append(hand_rect)
                else:
                    lhrect_list.append(hand_rect)

        else:
            lhand_rect = (0, 0, 0, 0)
            rhand_rect = (0, 0, 0, 0)
            lhrect_list.append(lhand_rect)
            rhrect_list.append(rhand_rect)

    return lhrect_list, rhrect_list


