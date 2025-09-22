import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

from autodistill_grounded_sam import GroundedSAM
# from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

import supervision as sv
from collections import Counter

from deep_sort_realtime.deepsort_tracker import DeepSort


class Detector():
    def __init__(self, caption_ontology, box_threshold=0.35, text_threshold=0.25) -> None:
        '''
        caption_ontology: dictionary of caption-label (prompt-label) pairs
            Ex) {
                    "bike": "Bike",
                    "person": "Person",
                    "helmet": "Helmet"
                }
        '''
        self.caption_ontology = caption_ontology
        self.base_model = GroundedSAM(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            ontology=CaptionOntology(caption_ontology)
        )
        # self.base_model = GroundedSAM2(           # nvcc required, which means CUDA bin file needed
        #     box_threshold=box_threshold,
        #     text_threshold=text_threshold,
        #     ontology=CaptionOntology(caption_ontology)
        # )
        print("Model loaded with the current caption ontology.")
        
        self.img_path = ''
        self.predictions = {}
        
        self.idx2label = {idx: label for idx, label in enumerate(caption_ontology.values())}
        # print(box_threshold, text_threshold)
        # print(self.idx2label)
    
    def predict(self, rgb_path) -> dict:
        self.img_path = rgb_path
        
        self.predictions = self.base_model.predict(rgb_path)
        
        res = {
            'metadata': self.predictions.metadata,
            'class_ids': self.predictions.class_id,
            'labels': [self.idx2label[class_id] for class_id in self.predictions.class_id],
            'boxes_xyxy': self.predictions.xyxy,
            'masks': self.predictions.mask,
            'conf': self.predictions.confidence,
            'track_ids': self.predictions.tracker_id,
            'data': self.predictions.data
        }
        
        return res

class PCSegmentationModel():
    def __init__(self, caption_ontology, box_threshold=0.35, text_threshold=0.25, to_save=True, save_dir='./tmp'):
        self.to_save = to_save
        self.save_dir = save_dir
        
        self.grounded_sam = Detector(caption_ontology, box_threshold=box_threshold, text_threshold=text_threshold)
    
    def detect(self, rgb_path):
        self.predictions = self.grounded_sam.predict(rgb_path)
        
        # Visualize the predictions
        num_preds = len(self.predictions['class_ids'])
        print(f"Number of detected objects: {num_preds}")
        for label in self.predictions['labels']:
            print(f"  - {label}")
        
        # Save the intermediate masks
        if self.to_save:
            for i, mask in enumerate(self.predictions['masks']):
                # save_dir = os.path.dirname(rgb_path)
                save_path = os.path.join(self.save_dir, f'mask_{i}_{label}.jpg')
                cv2.imwrite(save_path, mask.astype(int))
        # print(self.predictions['masks'][0].shape)

        return self.predictions
    
    def segment(self, pc):
        # Segment point clouds using the corresponding masks
        pcseg_list = []

        for mask in self.predictions['masks']:
            mask_flattened = np.array(mask).flatten()

            pcseg = pc[mask_flattened == True]
            pcseg_list.append(pcseg)
        
        # Save the resulting point cloud segments
        if self.to_save:
            for i, (pc_obj, label) in enumerate(zip(pcseg_list, self.predictions['labels'])):
                save_path = os.path.join(self.save_dir, f'pcseg_{i}_{label}.pcd')
                pc_obj.save(save_path)
        
        return range(len(pcseg_list)), self.predictions['labels'], pcseg_list



# class to combine both 
class DetectorTracker():
    def __init__(self, caption_ontology, box_threshold=0.35, text_threshold=0.25, max_age=30, nn_budget=70, nms_max_overlap=1.0, to_save=True, save_dir='./tmp'):
        self.to_save = to_save
        self.save_dir = save_dir
        
        self.model = None
        self.model_type = None
        self.class_labels = None
        self.tracker = None
        self.load_model(caption_ontology, box_threshold=box_threshold, text_threshold=text_threshold, max_age=max_age, nn_budget=nn_budget, nms_max_overlap=nms_max_overlap)

    def load_model(self, caption_ontology, box_threshold=0.35, text_threshold=0.25, max_age=30, nn_budget=70, nms_max_overlap=1.0):
        self.caption_ontology = caption_ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        self.model = Detector(caption_ontology, box_threshold=box_threshold, text_threshold=text_threshold)
        self.class_labels = list(self.caption_ontology.values())
        print('[load_model()] Model loaded, proceed with tracker')
        self.tracker = DeepSort(max_age=max_age, nn_budget=nn_budget, nms_max_overlap=nms_max_overlap)

    def update_tracks(self, detector_results, image_rgb):
        bounding_boxes = detector_results.get('boxes_xyxy')
        confidence_scores = detector_results.get('conf')
        class_ids = detector_results.get('class_ids')
        
        converted_boxes = []
        for box, confidence, class_id in zip(bounding_boxes, confidence_scores, class_ids):
            x1, y1, x2, y2 = [int(p) for p in box]
            width = x2 - x1
            height = y2 - y1
            converted_boxes.append([[x1, y1, width, height], confidence, class_id])
        
        tracks = self.tracker.update_tracks(converted_boxes, frame=image_rgb)
        return tracks
    
    def reset_tracks(self):
        return self.tracker.refresh_track_ids()
    
    def get_detections(self, rgb_path):
        self.detections = self.model.predict(rgb_path)
        return self.detections
    
    def get_tracks(self, detected_results, image_rgb):
        # return self.tracker.get_tracks(include_forbidden=include_forbidden)
        return self.update_tracks(detected_results, image_rgb)

    def __call__(self, image_rgb):
        return self.forward(image_rgb)
        
    def forward(self, image_rgb):
        # return self.tracker(cv2_image)
        return self.get_tracks(self.detections, image_rgb)

    # TODO: implement on the corresponding tracker
    def update_track_labels(self, track_ids, labels):
        # return self.tracker.update_track_labels(track_ids, labels)
        return self.tracker.update_track_ids()

