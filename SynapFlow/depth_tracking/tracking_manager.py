import numpy as np
import torch

class TrackingManager:
    def __init__(self, inactive_wait: int = 5):
        self.inactive_wait = inactive_wait
        self.active_tracks = {}
        self.inactive_tracks = {}

        self.next_new_id = 0
        self.boxes_idx_to_id = {} # one dict per frame

    def update_tracks(self, matches, cost_matrix, embeddings_t, boxes_t,
                      embeddings_t1, boxes_t1, im_id_t, im_id_t1, threshold):

        if self.next_new_id == 0:
            ids_t = np.arange(cost_matrix.shape[0])
            self.boxes_idx_to_id[im_id_t] = {
                i: i for i in range(cost_matrix.shape[0])
                }
            updated_active_tracks = {
                i: {
                    "embedding": embeddings_t[i],
                    "box": boxes_t[i],
                    "inactive_frames": 0
                } for i in range(cost_matrix.shape[0])}
        else:
            updated_active_tracks = {}
            # previous ids should already be registered as it is sequential
            ids_t = np.array(list(self.active_tracks.keys()))

        ids_t1 = -1 * np.ones(cost_matrix.shape[1])

        unmatched_t1 = np.arange(cost_matrix.shape[1]).tolist()

        boxes_idx_to_id_t1 = {}

        for t_idx, t1_idx in zip(*matches):
            if cost_matrix[t_idx, t1_idx] < threshold:
                track_id = ids_t[t_idx]
                ids_t1[t1_idx] = track_id
                updated_active_tracks[track_id] = {
                    "embedding": embeddings_t1[t1_idx],
                    "box": boxes_t1[t1_idx],
                    "inactive_frames": 0
                }
                # unmatched_t.remove(t_idx)
                unmatched_t1.remove(t1_idx)
                boxes_idx_to_id_t1[t1_idx] = track_id
        
        for track_id, track_data in self.active_tracks.items():
            if track_id not in updated_active_tracks:
                frames_since_last_seen = im_id_t1 - im_id_t
                track_data["inactive_frames"] += frames_since_last_seen
                if track_data["inactive_frames"] <= self.inactive_wait:
                    updated_active_tracks[track_id] = track_data
                else:
                    self.inactive_tracks[track_id] = track_data
        
        all_used_ids = list(updated_active_tracks.keys()) + list(self.inactive_tracks.keys())
        if len(all_used_ids) > 0:
            self.next_new_id = max(all_used_ids) + 1

        # new objects
        for t1_idx in unmatched_t1:
            updated_active_tracks[self.next_new_id] = {
                "embedding": embeddings_t1[t1_idx],
                "box": boxes_t1[t1_idx],
                "inactive_frames": 0
            }
            boxes_idx_to_id_t1[t1_idx] = self.next_new_id
            self.next_new_id += 1
        
        self.active_tracks = updated_active_tracks
        self.boxes_idx_to_id[im_id_t1] = boxes_idx_to_id_t1
    
    def get_boxes_idx_to_id(self, im_id):
        return self.boxes_idx_to_id[im_id]

    def get_active_tracks(self):
        return self.active_tracks
    
    def get_emb_box_active_tracks(self):
        active_tracks = self.get_active_tracks()
        embeddings = []
        boxes = []

        for track_id in active_tracks:
            embeddings.append(active_tracks[track_id]["embedding"])
            boxes.append(active_tracks[track_id]["box"])
        
        return torch.stack(embeddings), torch.stack(boxes)
    
    def get_box_active_tracks(self):
        active_tracks = self.get_active_tracks()
        boxes = []

        for track_id in active_tracks:
            boxes.append(active_tracks[track_id]["box"])
        
        return torch.stack(boxes)