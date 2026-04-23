# Technical Report: Multi-Object Detection & Persistent ID Tracking

**Pipeline:** YOLOv8m + BoT-SORT + K-Means Team Clustering  
**Footage:** Croatia vs Brazil — Handball World Championship (Quarterfinals)  
**Performance:** Processed 1548 frames | Avg tracked per frame: 18.7 | Peak simultaneous: 25

---

## 1. Objective and Approach
The goal of this assignment was to build a robust pipeline capable of detecting and persistently tracking multiple players in sports footage. Broadcast sports present unique challenges: extreme camera panning, heavy occlusion during group play, and players wearing identical kits. 

To solve this, I opted for a tracking-by-detection paradigm, combining a YOLOv8m detector with a BoT-SORT tracker, and layering an unsupervised K-Means clustering algorithm on top to separate the teams.

## 2. The Detector: YOLOv8m
I chose YOLOv8m (Medium) as the optimal balance between accuracy and compute efficiency for a Google Colab T4 environment. 
* **Why YOLOv8:** Its anchor-free architecture handles the massive scale variations in sports footage (e.g., small distant players vs. large close-up players) better than older YOLO versions.
* **Confidence Tuning (0.25):** I deliberately lowered the confidence threshold to 0.25. In wide aerial shots, players are often blurry or partially occluded. A lower threshold catches these edge cases, while the tracker naturally filters out brief false positives (like crowd members).

## 3. The Tracker: BoT-SORT (Customized)
While ByteTrack is a standard choice, I specifically went with BoT-SORT to handle the realities of broadcast handball footage.
* **Camera Motion Compensation (CMC):** Broadcast cameras constantly pan to follow the ball. BoT-SORT estimates the background movement and corrects the predicted bounding boxes before matching them. Without CMC, the tracker would interpret a camera pan as player movement, destroying ID consistency.
* **Fixing the Track Buffer (`track_buffer = 120`):** The default 30-frame memory in Ultralytics is too short. In handball, a player might run off-screen during a fast break or be blocked during a goalmouth scramble for several seconds. By increasing the buffer to 120 frames (4 seconds), the Kalman filter keeps predicting their position, allowing the pipeline to re-assign the correct ID when they reappear.

## 4. Optional Enhancement: Team Clustering
To handle the "similar-looking subjects" challenge, I implemented a K-Means clustering step. 
* Every 30 frames, the pipeline crops the upper 50% of each bounding box (isolating the jersey from the shorts/floor).
* It converts the crop to HSV color space (which is more resistant to stadium shadow/lighting changes than RGB) and extracts the mean color.
* K-Means (k=3) clusters the players into Team A (Croatia), Team B (Brazil), and the Referee. 

## 5. Challenges & Limitations
1. **Severe Occlusion:** During set plays or goalmouth scrambles, 4-5 players overlap completely. If the occlusion lasts longer than my 4-second buffer, an ID switch is inevitable. Integrating a dedicated Re-ID appearance model would be the next step to fix this.
2. **K-Means Flicker:** Occasionally, the referee (wearing dark colors) gets clustered with Brazil under certain lighting conditions. 

## 6. Conclusion
This pipeline successfully maintains ID consistency through standard sports movement and camera panning. By leveraging BoT-SORT's CMC and writing a custom 120-frame buffer, the system handles real-world broadcast noise effectively. Post-processing generates a Gaussian KDE position heatmap and a per-frame player count chart, both of which are included in the output.