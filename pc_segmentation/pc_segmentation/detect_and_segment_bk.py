import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

import supervision as sv
from collections import Counter


class Detector():
    def __init__(self, caption_ontology, box_threshold=0.4, text_threshold=0.4) -> None:
        '''
        caption_ontology: dictionary of caption-label pairs
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
        print("Model loaded with the current caption ontology.")
        
        self.img_path = ''
        self.predictions = {}
    
    def predict(self, rgb_path) -> dict:
        self.img_path = rgb_path
        
        self.predictions = self.base_model.predict(rgb_path)
        
        res = {
            'metadata': self.predictions.metadata,
            'class_ids': self.predictions.class_id,
            'tracker_ids': self.predictions.tracker_id,
            'boxes_xyxy': self.predictions.xyxy,
            'masks': self.predictions.mask
        }
        
        return res

    def plot_annotated_image(self):
        if self.img_path:
            print(self.img_path)
            
            image = cv2.imread(self.img_path)
            predictions = self.predictions

            annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_LEFT)
            mask_annotator = sv.MaskAnnotator()

            annotated_image = annotator.annotate(scene=image.copy(), detections=predictions)
            annotated_image = label_annotator.annotate(annotated_image, detections=predictions)
            annotated_image = mask_annotator.annotate(annotated_image, detections=predictions)
            
            # Count the occurrences of each class
            class_counts = Counter(predictions.class_id)
            class_names = {i: cap for i, cap in enumerate(self.caption_ontopology.keys())}
            
            # Create the title string
            title = ', '.join(f"{count} {class_names[class_id]}" for class_id, count in class_counts.items())
            
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            
            sv.plot_image(annotated_image, size=(4, 4))
            
        else:
            print("ERROR: No image given so far.")

class PCSegmentationModel():
    def __init__(self, caption_ontology, box_threshold=0.5, to_save=True):
        self.idx2label = {idx: label for idx, label in enumerate(caption_ontology.values())}
        self.to_save = to_save
        self.grounded_sam = Detector(caption_ontology, box_threshold=box_threshold)
    
    def detect(self, rgb_path):
        self.predictions = self.grounded_sam.predict(rgb_path)
        
        # Visualize the predictions
        num_preds = len(self.predictions['class_ids'])
        print(f"Number of detected objects: {num_preds}")
        for class_id in self.predictions['class_ids']:
            label = self.idx2label[class_id]
            print(f"  - {label}")
        
        # Save the intermediate masks
        if self.to_save:
            for i, mask in enumerate(self.predictions['masks']):
                save_dir = os.path.dirname(rgb_path)
                save_path = os.path.join(save_dir, f'mask_{i}_{label}.jpg')
                cv2.imwrite(save_path, mask.astype(int))

        return self.predictions
    
    # def segment(self, pc):
    #     # Segment point clouds using the corresponding masks
    #     pcseg_list = []

    #     for mask in self.predictions['masks']:
    #         mask_flattened = np.array(mask).flatten()

    #         pcseg = pc[mask_flattened == True]
    #         pcseg_list.append(pcseg)
        
    #     return pcseg_list

    def segment(self, pc):
        pcseg_list = []
        pc_points = pc.pc_data.shape[0]

        for mask in self.predictions['masks']:
            # Reshape mask to match point cloud size
            mask_flattened = cv2.resize(mask.astype(float), (pc_points, 1)).flatten()
            mask_bool = mask_flattened > 0.5
            pcseg = pc[mask_bool]
            pcseg_list.append(pcseg)
        
        return pcseg_list