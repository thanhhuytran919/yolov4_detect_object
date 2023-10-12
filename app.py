import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, make_response, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Đường dẫn đến thư mục chứa mô hình YOLOv4 và tệp trọng số
YOLOV4_PATH = "Model"

# Đường dẫn đến thư mục chứa ảnh đã detect có khung
STATIC_PATH = os.path.join(app.root_path, 'static')

# Thêm biến global để lưu thông tin detect
detected_info = None


def load_yolov4_model():
    config_path = os.path.join(YOLOV4_PATH, "yolov4-obj.cfg")
    weights_path = os.path.join(YOLOV4_PATH, "yolov4-obj_10000.weights")
    net = cv2.dnn.readNet(config_path, weights_path)

    # Đọc file obj.names và lưu danh sách tên lớp vào một danh sách
    names_file = os.path.join(YOLOV4_PATH, "obj.names")
    with open(names_file, "r") as file:
        class_names = [line.strip() for line in file.readlines()]

    return net, class_names


yolov4_model, class_names = load_yolov4_model()


def save_image(image, filename):
    output_path = os.path.join(STATIC_PATH, filename)
    cv2.imwrite(output_path, image)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    if request.method == 'POST':
        try:
            # Lấy tấm hình từ yêu cầu POST và thực hiện nhận dạng
            image = request.files['image']
            if image:
                uploaded_image_path = os.path.join(
                    STATIC_PATH, 'uploaded_image.jpg')
                image.save(uploaded_image_path)

                upload = f'static/uploaded_image.jpg'

                img = cv2.imread(uploaded_image_path)

                blob = cv2.dnn.blobFromImage(
                    img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                yolov4_model.setInput(blob)
                layer_names = yolov4_model.getUnconnectedOutLayersNames()
                detections = yolov4_model.forward(layer_names)

                max_confidence = 0
                best_detection = None
                for detection in detections:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5 and confidence > max_confidence:
                            max_confidence = confidence
                            best_detection = obj

                if best_detection is not None:
                    class_id = np.argmax(best_detection[5:])
                    confidence = best_detection[5 + class_id]
                    center_x = int(best_detection[0] * img.shape[1])
                    center_y = int(best_detection[1] * img.shape[0])
                    width = int(best_detection[2] * img.shape[1])
                    height = int(best_detection[3] * img.shape[0])
                    x = center_x - width // 2
                    y = center_y - height // 2

                    class_id = int(class_id)
                    x = int(x)
                    y = int(y)
                    width = int(width)
                    height = int(height)

                    class_name = class_names[class_id]

                    result = {
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                    }

                    img_detect = cv2.rectangle(
                        img, (x, y), (x + width, y + height), (32, 0, 198), 10)
                    font_scale = 2
                    text_color = (32, 0, 198)
                    text = f"{class_name} ({confidence*100:.2f}%)"
                    cv2.putText(img_detect, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 5)

                    detected_image_path = os.path.join(
                        STATIC_PATH, 'detected_image.jpg')
                    cv2.imwrite(detected_image_path, img_detect)
                    detect = f'static/detected_image.jpg'

                    return render_template(
                        'detect.html',
                        uploaded_image_path=upload,
                        detected_image_path=detect,
                        result_json=result,
                    )

                else:
                    return jsonify({'error': 'No objects detected.'})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return render_template('detect.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    try:
        # Lấy tấm hình từ yêu cầu POST và thực hiện nhận dạng
        image = request.files['image']
        if image:
            uploaded_image_path = os.path.join(
                STATIC_PATH, 'uploaded_image.jpg')
            image.save(uploaded_image_path)

            img = cv2.imread(uploaded_image_path)

            blob = cv2.dnn.blobFromImage(
                img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            yolov4_model.setInput(blob)
            layer_names = yolov4_model.getUnconnectedOutLayersNames()
            detections = yolov4_model.forward(layer_names)

            max_confidence = 0
            best_detection = None
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and confidence > max_confidence:
                        max_confidence = confidence
                        best_detection = obj

            if best_detection is not None:
                class_id = np.argmax(best_detection[5:])
                confidence = best_detection[5 + class_id]
                center_x = int(best_detection[0] * img.shape[1])
                center_y = int(best_detection[1] * img.shape[0])
                width = int(best_detection[2] * img.shape[1])
                height = int(best_detection[3] * img.shape[0])
                x = center_x - width // 2
                y = center_y - height // 2

                class_id = int(class_id)
                x = int(x)
                y = int(y)
                width = int(width)
                height = int(height)

                class_name = class_names[class_id]

                result = {
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                }

                img_detect = cv2.rectangle(
                    img, (x, y), (x + width, y + height), (32, 0, 198), 10)
                font_scale = 2
                text_color = (32, 0, 198)
                text = f"{class_name} ({confidence*100:.2f}%)"
                cv2.putText(img_detect, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 5)

                detected_image_path = os.path.join(
                    STATIC_PATH, 'detected_image.jpg')
                cv2.imwrite(detected_image_path, img_detect)

                return jsonify(result)
            else:
                return jsonify({'error': 'No objects detected.'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/download_json/<json_data>', methods=['GET'])
def download_json(json_data):
    response = make_response(json_data)
    response.headers["Content-Disposition"] = "attachment; filename=result.json"
    response.headers["Content-Type"] = "application/json"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
