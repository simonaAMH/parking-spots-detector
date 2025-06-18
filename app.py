from flask import Flask, jsonify
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

app = Flask(__name__)

def detect_parking_spots_occupancy(empty_img, current_img):
    # Step 1: Detect White Lines
    white_mask = detect_white_lines(empty_img)
    
    # Step 2: Detect Raw Line Segments
    lines = detect_line_segments(white_mask)
    
    # Step 3: Merge and Extend Lines to Form a Grid
    grid_lines = merge_and_extend_lines(lines, empty_img.shape)
    
    # Step 4: Generate and Filter Spots
    parking_spots = generate_spots_from_grid(grid_lines, empty_img.shape)
    filtered_spots = filter_invalid_spots(parking_spots)
    
    # Step 5: Determine Occupancy Status
    spots_with_status = determine_occupancy_status(empty_img, current_img, filtered_spots)
    
    # Return just the count of occupied spots
    occupied_count = sum(1 for spot in spots_with_status if spot['occupied'])
    total_spots = len(spots_with_status)
    
    return {
        'occupied_spots': occupied_count
    }

def determine_occupancy_status(empty_img, filled_img, parking_spots, threshold=0.03):
    spots_with_status = []
    
    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    filled_gray = cv2.cvtColor(filled_img, cv2.COLOR_BGR2GRAY)
    
    empty_blur = cv2.GaussianBlur(empty_gray, (5, 5), 0)
    filled_blur = cv2.GaussianBlur(filled_gray, (5, 5), 0)
    
    for i, (x, y, w, h) in enumerate(parking_spots):
        empty_roi = empty_blur[y:y+h, x:x+w]
        filled_roi = filled_blur[y:y+h, x:x+w]
        
        diff = cv2.absdiff(empty_roi, filled_roi)
        diff_normalized = diff.astype(np.float32) / 255.0
        change_percentage = np.mean(diff_normalized)
        
        is_occupied = change_percentage > threshold

        spot_info = {
            'id': int(i + 1),
            'occupied': bool(is_occupied),
            'change_percentage': float(change_percentage)
        }

        spots_with_status.append(spot_info)
    
    return spots_with_status

def detect_white_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((2, 1), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    kernel_small = np.ones((1, 1), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
    
    return white_mask

def detect_line_segments(mask):
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=25, maxLineGap=10)
    
    horizontal_lines, vertical_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 45 or angle > 135:
                horizontal_lines.append(line[0])
            else:
                vertical_lines.append(line[0])
    return {'horizontal': horizontal_lines, 'vertical': vertical_lines}

def merge_and_extend_lines(lines, img_shape):
    img_height, img_width, _ = img_shape
    merged_lines = {'horizontal': [], 'vertical': []}

    if lines['horizontal']:
        y_coords = np.array([(l[1] + l[3]) / 2 for l in lines['horizontal']]).reshape(-1, 1)
        db = DBSCAN(eps=15, min_samples=1).fit(y_coords)
        for label in set(db.labels_):
            cluster_lines = [lines['horizontal'][i] for i, l in enumerate(db.labels_) if l == label]
            avg_y = int(np.mean([(l[1] + l[3]) / 2 for l in cluster_lines]))
            merged_lines['horizontal'].append((0, avg_y, img_width, avg_y))

    if lines['vertical']:
        x_coords = np.array([(l[0] + l[2]) / 2 for l in lines['vertical']]).reshape(-1, 1)
        db = DBSCAN(eps=15, min_samples=1).fit(x_coords)
        for label in set(db.labels_):
            cluster_lines = [lines['vertical'][i] for i, l in enumerate(db.labels_) if l == label]
            avg_x = int(np.mean([(l[0] + l[2]) / 2 for l in cluster_lines]))
            merged_lines['vertical'].append((avg_x, 0, avg_x, img_height))
    
    return merged_lines

def generate_spots_from_grid(grid_lines, img_shape):
    spots = []
    horizontal = sorted(grid_lines['horizontal'], key=lambda l: l[1])
    vertical = sorted(grid_lines['vertical'], key=lambda l: l[0])
    img_height, img_width, _ = img_shape

    h_lines = [l[1] for l in horizontal]
    v_lines = [l[0] for l in vertical]
    h_lines = sorted(list(set([0] + h_lines + [img_height])))
    v_lines = sorted(list(set([0] + v_lines + [img_width])))
    
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i+1]
            x1, x2 = v_lines[j], v_lines[j+1]
            spots.append((x1, y1, x2 - x1, y2 - y1))
    return spots

def filter_invalid_spots(spots):
    filtered = []
    for x, y, w, h in spots:
        MIN_WIDTH, MAX_WIDTH = 40, 150
        MIN_HEIGHT, MAX_HEIGHT = 80, 250
        if (MIN_WIDTH < w < MAX_WIDTH) and (MIN_HEIGHT < h < MAX_HEIGHT):
            filtered.append((x, y, w, h))
    return filtered

@app.route('/detect-occupancy', methods=['GET'])
def detect_occupancy():
    try:
        empty_img = cv2.imread("empty_parking_lot.jpg")
        current_img = cv2.imread("not-empty-lot.png")
        result = detect_parking_spots_occupancy(empty_img, current_img)
        
        return jsonify({
            'success': True,
            'message': 'Occupancy detected successfully',
            **result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'message': f'Image processing error: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)