import os
import time
import cv2, torch

def cs_shot(predictor=None, args=None):
    import numpy as np
    from mss import mss
    from pynput import mouse, keyboard
    import win32api, win32gui,win32con, pyautogui

    def switch(key):
        key2 = keyboard.Key.alt.alt_l
        key1 = keyboard.Key.enter
        if key == key2:
            if off_on:
                off_on = False
            else: off_on = True

    # 320 256 192 128 64
    off_on = True
    mouse_con = mouse.Controller()
    half_scale = 192
    t0 = time.time()
    with keyboard.Listener(on_press=switch) as listener:
        hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')
        # rect = win32gui.GetWindowRect(hwnd)
        rect = (0,0,1920,1080)
        while 1:
            stat_xy = (0, 0)
            region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

            scale = 1920/region[2], 1080/region[3],

            frame = np.array(pyautogui.screenshot(region=region))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # cv2.imshow('1', frame)
            # cv2.waitKey()

            ih, iw = frame.shape[:2]
            t, l = int(ih/2-half_scale), int(iw/2-half_scale)
            img = frame[t:t+half_scale*2, l:l+half_scale*2].copy()

            cv2.rectangle(frame, (l, t), (l+half_scale*2, t+half_scale*2), (255, 255, 255), 2)

            outputs = [torch.tensor([[100, 100, 200, 200, 0.8, 0.9]])]
            img_info = {'ratio': 1}
            # outputs, img_info = predictor.inference(img, info_vis=False)
            # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre,show_conf=False)
            # frame[t:half_scale*2+t, l:half_scale*2+l] = cv2.resize(result_frame, (half_scale*2, half_scale*2))

            if outputs[0] != None:
                boxes = np.array(outputs[0].cpu())
                boxes[:, :4] = np.int16(boxes[:, :4]/img_info['ratio'])
                scores = boxes[:, 4] * boxes[:, 5]

                w_h = boxes[:, 2:4] - boxes[:, :2]
                # idx = np.argmax(w_h[:, 0] * w_h[:, 1])
                idx = np.argmax(scores)
                shot_x = int((boxes[idx, 0] + boxes[idx, 2])/2) + l + rect[0]
                shot_y = int((boxes[idx, 1] + boxes[idx, 3])/2 + t + rect[1])
                # shot_y = int((boxes[idx, 1] + boxes[idx, 3])/2 + t + rect[1] + w_h[idx, 1]*2)
                stat_xy = (shot_x, shot_y)

                if off_on:
                    org_xy = win32api.GetCursorPos()
                    move_xy = (stat_xy[0] - org_xy[0], stat_xy[1] - org_xy[1])
                    # move_xy = int(move_xy[0]*scale[0]), int(move_xy[1]*scale[1])
                    move_d = (move_xy[0] ** 2 + move_xy[1] ** 2) ** 0.5

                    if stat_xy != (0, 0) and move_d < half_scale:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_xy[0], move_xy[1], 0, 0)
                        # pyautogui.press()
                        # print(f'move:', org_xy, stat_xy, move_xy, move_d, '\n')
                    # else:
                    #     print(f'not move:', org_xy, stat_xy, move_xy, move_d, '\n')

                cv2.circle(frame, (shot_x, shot_y), 2, (0, 255, 0), 3)

                for i, box in enumerate(boxes):
                    if scores[i]>0.5:
                        cv2.rectangle(frame,
                        (int(box[0] + l), int(box[1] + t)),
                        (int(box[2] + l), int(box[3] + t)), (0, 255, 0), 2)

            t = time.time()
            # print('{:.1f} FPS'.format(1 / (t - t0)))
            cv2.putText(frame, '{:.1f} FPS'.format(1 / (t - t0)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=4)
            cv2.imshow("shot", cv2.resize(frame, (960, 540)))
            cv2.waitKey(1)
            t0 = time.time()

if __name__ =="__main__":
    cs_shot()