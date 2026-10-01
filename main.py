import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
parking_spots = {
    "Spot_1": [307, 316, 330, 360],
    "Spot_2": [333, 321, 354, 358],
    "Spot_3": [360, 326, 388, 355],
    "Spot_4": [443, 286, 465, 313],
    "Spot_5": [417, 320, 439, 348],
    "Spot_6": [25, 160, 46, 229],
    "Spot_7": [55, 169, 81, 220],
    "Spot_8": [81, 172, 106, 221],
    "Spot_9": [178, 189, 197, 235],
    "Spot_10": [202, 188, 229, 230],
    "Spot_11": [146, 177, 167, 226],
    "Spot_12": [114, 177, 131, 222],
    "Spot_13": [2, 168, 20, 208],
    "Spot_14": [471, 284, 487, 310],
    "Spot_15": [502, 283, 510, 305],
    "Spot_16": [508, 323, 520, 347],
    "Spot_17": [475, 322, 496, 345],
    "Spot_18": [632, 416, 657, 444],
    "Spot_19": [677, 415, 698, 441],
    "Spot_20": [724, 407, 745, 431],
    "Spot_21": [760, 407, 785, 432],
    "Spot_22": [803, 403, 824, 428],
    "Spot_23": [849, 395, 870, 424],
    "Spot_24": [888, 398, 911, 420],
    "Spot_25": [936, 389, 959, 418],
    "Spot_26": [975, 389, 1003, 418],
    "Spot_27": [14, 400, 24, 439],
    "Spot_28": [41, 416, 47, 450],
    "Spot_29": [67, 411, 80, 450],
    "Spot_30": [97, 406, 119, 450],
    "Spot_31": [134, 412, 141, 454],
    "Spot_32": [165, 416, 184, 455],
    "Spot_33": [198, 433, 220, 467],
    "Spot_34": [239, 434, 260, 463],
    "Spot_35": [285, 433, 310, 463],
    "Spot_36": [336, 424, 366, 462],
    "Spot_37": [379, 435, 400, 461],
    "Spot_38": [449, 321, 461, 352],
    "Spot_39": [425, 423, 460, 466],
    "Spot_40": [469, 429, 493, 457],
    "Spot_41": [212, 282, 221, 311],
    "Spot_42": [186, 326, 201, 345],
    "Spot_43": [161, 321, 171, 350],
    "Spot_44": [186, 286, 192, 310],
    "Spot_45": [157, 285, 174, 305],
    "Spot_46": [141, 324, 148, 348],
    "Spot_47": [144, 284, 150, 310],
    "Spot_48": [140, 290, 148, 306],
    "Spot_49": [18, 313, 22, 333],
    "Spot_50": [64, 317, 68, 332],
    "Spot_51": [422, 292, 436, 312],
    "Spot_52": [372, 226, 383, 244],
    "Spot_53": [398, 185, 409, 201],
    "Spot_54": [422, 222, 432, 239],
    "Spot_55": [442, 220, 455, 235],
    "Spot_56": [467, 220, 477, 233],
    "Spot_57": [492, 218, 502, 235],
    "Spot_58": [544, 281, 561, 302],
    "Spot_59": [550, 317, 568, 338],
    "Spot_60": [573, 280, 585, 298],
    "Spot_61": [582, 317, 593, 335],
    "Spot_62": [598, 283, 610, 305],
    "Spot_63": [608, 314, 622, 336],
    "Spot_64": [625, 278, 640, 299],
    "Spot_65": [633, 313, 646, 328],
    "Spot_66": [646, 275, 664, 298],
    "Spot_67": [658, 310, 669, 332],
    "Spot_68": [674, 276, 681, 297],
    "Spot_69": [685, 310, 699, 330],
    "Spot_70": [696, 274, 711, 291],
    "Spot_71": [710, 307, 728, 328],
    "Spot_72": [724, 268, 737, 290],
    "Spot_73": [740, 306, 752, 323],
    "Spot_74": [745, 263, 756, 290],
    "Spot_75": [766, 305, 781, 322],
    "Spot_76": [768, 268, 786, 286],
    "Spot_77": [789, 298, 802, 320],
    "Spot_78": [795, 262, 809, 282],
    "Spot_79": [817, 298, 830, 322],
    "Spot_80": [821, 265, 836, 287],
    "Spot_81": [845, 294, 857, 316],
    "Spot_82": [846, 260, 860, 282],
    "Spot_83": [866, 295, 880, 315],
    "Spot_84": [871, 260, 880, 277],
    "Spot_85": [895, 258, 911, 271],
    "Spot_86": [912, 285, 932, 306],
    "Spot_87": [890, 293, 906, 310],
}


# ========================================================
# 3. OCCUPANCY MEMORY
# ========================================================

occupied_frames = {
    spot: 0 for spot in parking_spots
}

empty_frames = {
    spot: 0 for spot in parking_spots
}

spot_status = {
    spot: False for spot in parking_spots
}


# Number of consecutive frames required
OCCUPIED_CONFIRM_FRAMES = 3
EMPTY_CONFIRM_FRAMES = 5


# ========================================================
# 4. OPEN VIDEO
# ========================================================

video = cv2.VideoCapture("sample_video.mp4")

if not video.isOpened():
    print("ERROR: Could not open sample_video.mp4")
    exit()


# ========================================================
# 5. CREATE WINDOWS
# ========================================================

window_name = "Parking Camera"
dashboard_name = "Parking Dashboard"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(dashboard_name, cv2.WINDOW_NORMAL)

cv2.resizeWindow(window_name, 1020, 500)
cv2.resizeWindow(dashboard_name, 400, 500)


# ========================================================
# 6. MAIN LOOP
# ========================================================

while True:

    success, frame = video.read()

    if not success:
        print("Video finished playing!")
        break


    # ====================================================
    # RESIZE VIDEO
    # ====================================================

    frame = cv2.resize(
        frame,
        (1020, 500)
    )


    # ====================================================
    # 7. YOLO VEHICLE DETECTION
    # ====================================================

    results = model(
        frame,
        conf=0.02,
        iou=0.40,
        classes=[2, 5, 7],
        imgsz=1024
    )


    car_boxes = []


    # ====================================================
    # 8. DRAW DETECTED VEHICLES
    # ====================================================

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            car_boxes.append(
                [x1, y1, x2, y2]
            )


            # Blue vehicle bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )


    # ====================================================
    # 9. PARKING SPACE ANALYSIS
    # ====================================================

    total_spots = len(parking_spots)

    empty_count = 0


    for spot_name, spot_coords in parking_spots.items():

        sx1, sy1, sx2, sy2 = spot_coords


        # -----------------------------------------------
        # Parking spot center
        # -----------------------------------------------

        spot_center_x = (
            sx1 + sx2
        ) // 2

        spot_center_y = (
            sy1 + sy2
        ) // 2


        # -----------------------------------------------
        # Parking spot area
        # -----------------------------------------------

        spot_area = (
            (sx2 - sx1)
            *
            (sy2 - sy1)
        )


        current_occupied = False


        # =================================================
        # CHECK EVERY VEHICLE
        # =================================================

        for car in car_boxes:

            cx1, cy1, cx2, cy2 = car


            # ---------------------------------------------
            # METHOD 1:
            # Parking spot center inside vehicle
            # ---------------------------------------------

            center_inside_car = (
                cx1 <= spot_center_x <= cx2
                and
                cy1 <= spot_center_y <= cy2
            )


            # ---------------------------------------------
            # METHOD 2:
            # Rectangle overlap
            # ---------------------------------------------

            ix1 = max(
                sx1,
                cx1
            )

            iy1 = max(
                sy1,
                cy1
            )

            ix2 = min(
                sx2,
                cx2
            )

            iy2 = min(
                sy2,
                cy2
            )


            intersection_width = max(
                0,
                ix2 - ix1
            )

            intersection_height = max(
                0,
                iy2 - iy1
            )


            overlap_area = (
                intersection_width
                *
                intersection_height
            )


            if spot_area > 0:

                overlap_percentage = (
                    overlap_area /
                    spot_area
                ) * 100

            else:

                overlap_percentage = 0


            # ---------------------------------------------
            # OCCUPIED DECISION
            # ---------------------------------------------

            if (
                center_inside_car
                or
                overlap_percentage > 8
            ):

                current_occupied = True
                break


        # =================================================
        # 10. TEMPORAL CONFIRMATION
        # =================================================

        if current_occupied:

            occupied_frames[spot_name] += 1
            empty_frames[spot_name] = 0

        else:

            empty_frames[spot_name] += 1
            occupied_frames[spot_name] = 0


        # Confirm occupied
        if (
            occupied_frames[spot_name]
            >= OCCUPIED_CONFIRM_FRAMES
        ):

            spot_status[spot_name] = True


        # Confirm empty
        if (
            empty_frames[spot_name]
            >= EMPTY_CONFIRM_FRAMES
        ):

            spot_status[spot_name] = False


        # =================================================
        # 11. DRAW PARKING SPACE
        # =================================================

        if spot_status[spot_name]:

            # RED = OCCUPIED

            cv2.rectangle(
                frame,
                (sx1, sy1),
                (sx2, sy2),
                (0, 0, 255),
                2
            )

        else:

            # GREEN = AVAILABLE

            empty_count += 1

            cv2.rectangle(
                frame,
                (sx1, sy1),
                (sx2, sy2),
                (0, 255, 0),
                2
            )


    # ====================================================
    # 12. CALCULATE STATISTICS
    # ====================================================

    occupied_count = (
        total_spots -
        empty_count
    )


    occupancy_percentage = (
        occupied_count /
        total_spots
    ) * 100


    # ====================================================
    # 13. CREATE SEPARATE DASHBOARD
    # ====================================================

    dashboard = np.ones(
        (500, 400, 3),
        dtype=np.uint8
    ) * 255


    # ====================================================
    # DASHBOARD BORDER
    # ====================================================

    cv2.rectangle(
        dashboard,
        (5, 5),
        (395, 495),
        (40, 40, 40),
        2
    )


    # ====================================================
    # TITLE
    # ====================================================

    cv2.putText(
        dashboard,
        "YOLO PARKING",
        (85, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2
    )

    cv2.putText(
        dashboard,
        "MONITOR",
        (105, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )


    # ====================================================
    # AVAILABLE SECTION
    # ====================================================

    cv2.putText(
        dashboard,
        "AVAILABLE",
        (115, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 140, 0),
        2
    )

    cv2.putText(
        dashboard,
        str(empty_count),
        (165, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 140, 0),
        4
    )


    # ====================================================
    # OCCUPIED SECTION
    # ====================================================

    cv2.putText(
        dashboard,
        "OCCUPIED",
        (125, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 200),
        2
    )

    cv2.putText(
        dashboard,
        str(occupied_count),
        (165, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 200),
        4
    )


    # ====================================================
    # TOTAL SPACES
    # ====================================================

    cv2.putText(
        dashboard,
        "TOTAL SPACES",
        (105, 365),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 0),
        2
    )

    cv2.putText(
        dashboard,
        str(total_spots),
        (170, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3
    )


    # ====================================================
    # OCCUPANCY PERCENTAGE
    # ====================================================

    cv2.putText(
        dashboard,
        f"Occupancy: {occupancy_percentage:.1f}%",
        (75, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )


    # ====================================================
    # 14. SHOW WINDOWS
    # ====================================================

    cv2.imshow(
        window_name,
        frame
    )

    cv2.imshow(
        dashboard_name,
        dashboard
    )


    # ====================================================
    # 15. PRESS Q TO EXIT
    # ====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ========================================================
# 16. FINAL PARKING SUMMARY
# ========================================================

print()
print("========================================")
print("        PARKING ANALYSIS SUMMARY")
print("========================================")

print(f"Total Parking Spaces : {total_spots}")
print(f"Occupied Spaces      : {occupied_count}")
print(f"Available Spaces     : {empty_count}")
print(f"Occupancy Rate       : {occupancy_percentage:.1f}%")

print("========================================")


# ========================================================
# 17. CLEANUP
# ========================================================

video.release()
cv2.destroyAllWindows()
