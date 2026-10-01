import cv2
# 1. OPEN THE VIDEO AND GRAB THE FIRST FRAME
video = cv2.VideoCapture('sample_video.mp4')
success, frame = video.read()
if not success:
    print(" ERROR: Could not find 'sample_video.mp4'. Make sure it's in your folder!")
else:
    # Resize to our standard display size
    frame = cv2.resize(frame, (1020, 500))

    print("\n-----------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. CLICK and DRAG your mouse to draw a box around ONE parking spot.")
    print("2. Press ENTER or SPACE to lock it in.")
    print("3. Repeat for 3 to 5 different spots.")
    print("4. Press 'q' on your keyboard when you are totally finished.")
    print("-----------------------------------------------------\n")

    # 2. OPEN THE CLICK-AND-DRAG MULTI-BOX TOOL
    rois = cv2.selectROIs("Drag Boxes Over Straight Spots (Press Q when done)", frame, fromCenter=False, showCrosshair=True)

    # 3. GENERATE YOUR CUSTOM COORDINATES
    print("\n DONE! COPY AND PASTE THIS EXACT TEXT INTO YOUR MAIN.PY FILE:\n")
    print("parking_spots = {")
    for i, box in enumerate(rois):
        x, y, w, h = box
        x2 = x + w
        y2 = y + h
        print(f'    "Spot_{i+1}": [{x}, {y}, {x2}, {y2}],')
    print("}")
cv2.destroyAllWindows()