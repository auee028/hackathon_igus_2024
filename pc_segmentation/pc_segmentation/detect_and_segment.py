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


class Detector():
    def __init__(self, caption_ontology, box_threshold=0.35, text_threshold=0.25) -> None:
        '''
        caption_ontology: dictionary of label-caption pairs
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
            'tracker_ids': self.predictions.tracker_id,
            'boxes_xyxy': self.predictions.xyxy,
            'masks': self.predictions.mask,
            'labels': [self.idx2label[class_id] for class_id in self.predictions.class_id]
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
