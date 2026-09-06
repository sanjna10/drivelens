import json

import keras_cv
import tensorflow as tf

from config import (
    ANNOTATION_FILE,
    DETECTION_IMAGE_SIZE,
    DETECTOR_WEIGHTS_FILE,
)


class ObjectDetector:
    """RetinaNet + ResNet-50 object detector used to verify retrieved scenes."""

    def __init__(self):
        with open(ANNOTATION_FILE, "r") as f:
            coco = json.load(f)

        self.class_mapping = {
            category["id"]: category["name"]
            for category in coco["categories"]
        }

        num_classes = len(coco["categories"])

        self.detector = keras_cv.models.RetinaNet.from_preset(
            "resnet50_imagenet",
            num_classes=num_classes,
            bounding_box_format="xywh",
        )

        self.detector.load_weights(str(DETECTOR_WEIGHTS_FILE))

    def prepare_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

        original_shape = tf.shape(image)

        image = tf.image.resize(
            image,
            DETECTION_IMAGE_SIZE,
        )

        return image, original_shape

    def detect_objects(self, image_path, confidence_threshold=0.5):
        image, _ = self.prepare_image(image_path)

        batch = tf.expand_dims(image, axis=0)

        predictions = self.detector.predict(
            batch,
            verbose=0,
        )

        boxes = predictions["boxes"][0]
        classes = predictions["classes"][0]
        confidence = predictions["confidence"][0]

        detections = []

        for box, cls, score in zip(boxes, classes, confidence):
            score = float(score)

            if score < confidence_threshold:
                continue

            class_id = int(cls)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_mapping.get(
                        class_id,
                        str(class_id),
                    ),
                    "confidence": score,
                    "bbox": box.tolist(),
                }
            )

        return detections
