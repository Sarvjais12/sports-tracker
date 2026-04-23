# 🏆 Multi-Object Detection & Persistent ID Tracking

A robust computer vision pipeline utilizing YOLOv8m and BoT-SORT to detect and persistently track players in public sports footage. Features include team clustering, trajectory trails, and automated analytics.

### 🔗 Quick Links
* **Original Video Source:** https://www.youtube.com/watch?v=qBytWpdbTsk&t=392s
* **Demo Walkthrough Video:** [INSERT YOUR GOOGLE DRIVE DEMO LINK HERE]

---

## 🎯 Pipeline Architecture
1. **Detection:** `YOLOv8m` detects all persons (class 0, conf ≥ 0.25).
2. **Tracking:** `BoT-SORT` assigns persistent IDs using a Kalman filter and Camera Motion Compensation. (Custom `track_buffer = 120` to handle long occlusions).
3. **Team Clustering:** Unsupervised `K-Means (k=3)` on HSV color space separates Team A, Team B, and the Referee based on jersey color.
4. **Visual Analytics:** Generates trajectory trails, a tactical mini-map, and live active player stats.

## 📁 Repository Structure
* `MultiObject_Tracking_Assignment.ipynb`: Main execution code.
* `custom_sports_tracker.yaml`: Contains the 120-frame buffer fix.
* `requirements.txt`: List of dependencies.
* `report.md`: Technical write-up of architecture and limitations.
* `/output`: Contains the annotated video and generated heatmaps.
* `/screenshots`: Sample frames of the tracking in action.

## 🚀 How to Run (Google Colab)
1. Open the `.ipynb` file in Google Colab.
2. Set Runtime to **T4 GPU**.
3. Upload your test video to the `/content/` directory.
4. Run all cells sequentially. The required libraries (`ultralytics`, `scikit-learn`, `matplotlib`) will install automatically via the first cell.
5. Download output files using the final cell (tracked video, heatmap, count plot).