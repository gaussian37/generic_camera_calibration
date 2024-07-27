import os
from datetime import datetime
import cv2
import numpy as np
import csv
import argparse

harris_corners = []
clicked_points = []
clone = None

def draw_points(image):
    global clicked_points
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    color = (0, 0, 255)
    thickness = 1
    lineType = cv2.LINE_AA

    for index, point in enumerate(clicked_points):
        org = (point[1], point[0] - 10)
        cv2.putText(image, str(index+1).zfill(3), org, fontFace, fontScale, color, thickness, lineType)
        cv2.circle(image, (point[1], point[0]), 3, (255, 0, 0), thickness = 1)
        cv2.circle(image, (point[1], point[0]), 1, (0, 0, 255), thickness = -1)
    return image

def find_nearest_corner(centroids, given_point):
    # Convert to numpy array for easier manipulation
    centroids = np.array(centroids)
    given_point = np.array(given_point)

    # Calculate Euclidean distance from given_point to each corner point
    distances = np.linalg.norm(centroids - given_point, axis=1)

    # Find the index of the minimum distance
    nearest_index = np.argmin(distances)
    nearest_corner = centroids[nearest_index]

    return nearest_corner

def MouseEvents(event, x, y, flags, param):   
    
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        # Find the checkerboard corners
        nearest_corner = find_nearest_corner(harris_corners[:, :2], (x, y))
        nearest_corner = [int(nearest_corner[0]), int(nearest_corner[1])]
        clicked_points.append([nearest_corner[1], nearest_corner[0]])
        print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {nearest_corner[0]}, {nearest_corner[1]}")
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([y, x])
        print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")

    if event == cv2.EVENT_LBUTTONDOWN:
        image = clone.copy()
        image = draw_points(image)
        cv2.imshow("image", image)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', required=True)
parser.add_argument('--save_path', default="./results")
parser.add_argument('--resize_ratio', type=float, default=1.0)
args = parser.parse_args()

def main():
    global clone, clicked_points, harris_corners

    now = datetime.now()
    now_str = "%s%02d%02d_%02d%02d%02d" % (now.year - 2000, now.month, now.day, now.hour, now.minute, now.second)   

    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (int(image.shape[1] * args.resize_ratio), int(image.shape[0] * args.resize_ratio)))
    clone = image.copy()
    
    # Harris corner detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

    # Dilate corner image to enhance corner points
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # Find centroids of the corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    harris_corners = centroids

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseEvents)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(0)

        if key == 32: # space
            for i in range(len(clicked_points)):
                clicked_points[i][0] = int(clicked_points[i][0] * (1/args.resize_ratio))
                clicked_points[i][1] = int(clicked_points[i][1] * (1/args.resize_ratio))
                
            save_path = args.save_path.rstrip(os.sep) + os.sep + os.path.basename(args.image_path).rstrip(".png") + "_" + now_str
            os.makedirs(save_path, exist_ok=True)
            with open(save_path + os.sep + 'points.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['no', 'u', 'v'])
                # Write the data rows
                for index, point in enumerate(clicked_points):
                    writer.writerow( (index+1, point[1], point[0]) )
                    
            if len(clicked_points) > 0:
                image = cv2.imread(args.image_path)
                image = draw_points(image)
                cv2.imwrite(save_path + os.sep + "points.png", image)
            break

        if key == 8 or  key == ord('b'):  # backspace or b(back)
            if len(clicked_points) > 0:
                clicked_points.pop()
                image = clone.copy()
                image = draw_points(image)
                cv2.imshow("image", image)
        
        elif key == ord('w'):
            if len(clicked_points) > 0:
                if clicked_points[-1][0] > 0:
                    clicked_points[-1][0] -= 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)

        elif key == ord('s'):
            if len(clicked_points) > 0:
                if clicked_points[-1][0] < image.shape[0]-1:
                    clicked_points[-1][0] += 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)

        elif key == ord('a'):
            if len(clicked_points) > 0:
                if clicked_points[-1][1] > 0:
                    clicked_points[-1][1] -= 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)

        elif key == ord('d'):
            if len(clicked_points) > 0:
                if clicked_points[-1][1] < image.shape[1]-1:
                    clicked_points[-1][1] += 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)
        else:
            image = clone.copy()
            image = draw_points(image)
            cv2.imshow("image", image)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()