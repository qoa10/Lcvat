* OS: **Windows 10/11**
* We use **Ubuntu in WSL2**
* We use official **CVAT** repo + built-in **Nuclio serverless** + **SAM (pytorch/facebookresearch/sam)**

---

# Local CVAT + SAM Setup Guide (Windows + WSL2)

## 0. Prerequisites

1. **Windows 10 / 11** with admin rights
2. **Internet connection** (images will be downloaded by Docker)
3. Enough disk space (a few GB for Docker images)

---

## 1. Install WSL2 + Ubuntu

1. Open **PowerShell (Run as Administrator)** and run:

   ```powershell
   wsl --install
   ```

2. When prompted, choose **Ubuntu** as the distribution.

3. After installation, Windows will open an **Ubuntu** terminal and ask you to:

   * Create a **username** (e.g. `student1`)
   * Create a **password**

Later you can open Ubuntu anytime from Start menu: **“Ubuntu”**.

---

## 2. Install Docker Desktop (with WSL integration)

1. Download **Docker Desktop for Windows (AMD64)** from the Docker website.
2. Run the installer (ask your supervisor to type the admin password if needed).
3. During installation, **enable “Use WSL2 instead of Hyper-V”**.
4. After installation, open **Docker Desktop**.
5. Go to **Settings → Resources → WSL integration**:

   * Turn on **“Use Docker Desktop with WSL2”**
   * Enable Docker integration for **Ubuntu**.
6. Make sure Docker Desktop is running (icon in system tray).

---

## 3. Clone CVAT in Ubuntu (WSL)

1. Open **Ubuntu**.

2. Go to a folder on your Windows drive, for example:

   ```bash
   cd /mnt/f/Tao   # or another folder you like
   ```

3. Clone CVAT (or your teaching fork):

   ```bash
   git clone https://github.com/opencv/cvat.git
   cd cvat
   ```

4. The main docker files live here (we will call this **`cvat_src`**):

   ```bash
   ls
   # you should see: docker-compose.yml, components/, serverless/, ...
   ```

---

## 4. Start CVAT **with serverless enabled**

> ⚠️ Important: once you use serverless, **always** start CVAT with **two** compose files.

From inside the CVAT folder:

```bash
cd /mnt/f/Tao/cvat   # adjust path if different

docker compose \
  -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

* This starts:

  * CVAT backend + UI on **port 8080**
  * Nuclio dashboard on **port 8070**
  * Databases and workers

Check that containers are running:

```bash
docker ps
```

You should see `cvat_server`, `cvat_ui`, `nuclio`, etc.

---

## 5. Create CVAT admin user

Run in Ubuntu:

```bash
cd /mnt/f/Tao/cvat

docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
```

Follow the prompts, e.g.:

* Username: `tao`
* Email: `xxx@xxx.com`
* Password: `<your password>`

---

## 6. Log in to CVAT & Nuclio

* Open **CVAT UI** in browser:
  👉 [http://localhost:8080](http://localhost:8080)
  Log in with the admin account you just created.

* Open **Nuclio dashboard** (for debugging / viewing functions):
  👉 [http://localhost:8070](http://localhost:8070)

If 8070 doesn’t open, in Ubuntu check:

```bash
docker ps | grep -i nuclio
docker logs --tail 80 nuclio
```

---

## 7. Deploy SAM (CPU version – recommended first)

1. In Ubuntu, go to the `serverless` folder:

   ```bash
   cd /mnt/f/Tao/cvat/serverless
   ```

2. Deploy SAM **CPU** function:

   ```bash
   ./deploy_cpu.sh pytorch/facebookresearch/sam/
   ```

   * The script prints logs while it:

     * Creates project `cvat` in Nuclio
     * Builds image `cvat.pth.facebookresearch.sam.vit_h`
   * It may take a while; don’t interrupt even if you see `(W)` warnings.

3. In a **second** Ubuntu terminal you can monitor progress:

   ```bash
   nuctl get functions -n nuclio | grep -i sam
   ```

   When it’s done, you should see something like:

   ```text
   nuclio | pth-facebookresearch-sam-vit-h | cvat | ready | 1/1
   ```

4. You can also check the image:

   ```bash
   docker images --format "{{.Repository}}:{{.Tag}}" | grep -i sam
   ```

---

## 8. (Optional) Deploy SAM GPU version

> Do this **only after CPU is working**, and only on machines with a supported NVIDIA GPU + drivers.

From `serverless`:

```bash
cd /mnt/f/Tao/cvat/serverless
./deploy_gpu.sh pytorch/facebookresearch/sam/
```

* Same as before: watch with

  ```bash
  nuctl get functions -n nuclio | grep -i sam
  docker logs --tail 200 nuclio
  ```

* **Don’t stop the script** just because you see:

  ```text
  (W) Using user provided base image {"baseImage": "ubuntu:22.04"}
  ```

  This is *only a warning*, not an error.

---

## 9. Verify SAM is visible in CVAT

1. Open **CVAT**: [http://localhost:8080](http://localhost:8080)

2. Log in as admin.

3. Go to **Models →** you should see a model similar to:

   * `pth-facebookresearch-sam-vit-h`

4. Now in a task, you can use:

   * **Automatic Annotation** or **Interactive Segmentation** with the SAM model.

---

## 10. How to stop and restart CVAT properly

### Stop everything

From the CVAT folder:

```bash
cd /mnt/f/Tao/cvat
docker compose down --remove-orphans
```

### Start again (with serverless!)

```bash
docker compose \
  -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

> 🔴 **Do NOT** run `docker compose up -d` alone after using serverless.
> It can start a broken Nuclio container and make SAM functions get stuck in `building`.

---

## 11. Quick troubleshooting cheatsheet

### A. Nuclio container is `Restarting`

```bash
docker ps | grep -i nuclio
docker logs --tail 120 nuclio
```

If it keeps restarting:

```bash
cd /mnt/f/Tao/cvat
docker compose down --remove-orphans
docker rm -f nuclio 2>/dev/null || true

docker compose \
  -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d --force-recreate
```

---

### B. SAM function stuck in `building`

Check:

```bash
nuctl get functions -n nuclio | grep -i sam
```

If it never moves past `building`:

1. Delete and recreate the project:

   ```bash
   nuctl delete project cvat -n nuclio 2>/dev/null || true
   nuctl create project cvat -n nuclio
   ```

2. Redeploy SAM (start with CPU):

   ```bash
   cd /mnt/f/Tao/cvat/serverless
   ./deploy_cpu.sh pytorch/facebookresearch/sam/
   ```

---

### C. Useful commands list (for students)

```bash
# show containers
docker ps

# show only nuclio container
docker ps | grep -i nuclio

# show nuclio logs
docker logs --tail 120 nuclio

# show nuclio functions
nuctl get functions -n nuclio

# show SAM-related images
docker images --format "{{.Repository}}:{{.Tag}}" | grep -i sam

# start CVAT + serverless
cd /mnt/f/Tao/cvat
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

# stop CVAT
docker compose down --remove-orphans
```

# # 📘 **End-to-End Workflow Summary (English + Code Steps)**

This workflow converts **YOLO bounding boxes → SAM2 segmentation masks → YOLO-seg polygons → COCO JSON → CVAT editable annotations**.

---

# ## 1. **Activate the project environment**

Activate your Python virtual environment before running any scripts.

```bash
cd F:\Tao\cvat
.\.venv310\Scripts\activate
```

---

# ## 2. **Start CVAT + Serverless backend (Nuclio)**

This brings up CVAT with all required containers.

```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

---

# ## 3. **Verify that CVAT is running**

Use Docker to confirm that CVAT containers are alive.

```bash
docker ps
```

---

# ## 4. **Run SAM2 on a single image + YOLO box file**

This generates a segmentation mask + YOLO-seg polygons + visualization.

```bash
python box2mask_sam2.py --image "path/to/img.png" --yolo-txt "path/to/label.txt" --out "output_mask.png" --model-id "facebook/sam2-hiera-large"
```

---

# ## 5. **Run SAM2 on an entire folder (batch processing)**

This converts all YOLO bounding-box labels into YOLO-seg polygon labels.

```bash
python box2mask_sam2.py --images-dir "F:\Tao\dataset\images" --labels-dir "F:\Tao\dataset\labels" --out-dir "F:\Tao\dataset\labels_seg" --model-id "facebook/sam2-hiera-large"
```

# ## 6. **Run SAM2 create**

This converts all YOLO bounding-box labels into YOLO-seg polygon labels.

```bash
 python F:\CrackDetection_v2.2.3\sam2seg\code\yolo_sam2_to_yoloseg.py `
>>   --images-dir "F:\CrackDetection_v2.2.3\sam2seg\data\acp_subset\images\val" `
>>   --det-labels-dir "F:\CrackDetection_v2.2.3\sam2seg\data\acp_subset\labels\val" `
>>   --seg-labels-dir "F:\CrackDetection_v2.2.3\sam2seg\data\acp_subset\labels_seg\val" `
>>   --debug-vis-dir "F:\CrackDetection_v2.2.3\sam2seg\data\acp_subset\debug_vis\val" `
>>   --model-id "facebook/sam2-hiera-large"

---

# ## 6. **Convert YOLO-seg polygon labels → COCO format for CVAT**

This creates a CVAT-compatible JSON file.

```bash
python yoloseg_to_coco_cvat.py --images-dir "F:\Tao\cvat\test\images" --seg-labels-dir "F:\Tao\cvat\test\labels_seg" --output-json "F:\Tao\cvat\test\coco_sam2_for_cvat.json"
```

---

# ## 7. **Open CVAT in the browser**

Access the CVAT interface to create tasks and import annotations.

```text
http://localhost:8080
```

---

# ## 8. **Create a new CVAT task**

Add labels matching your dataset classes (e.g., class_0 … class_n).

```text
CVAT → Tasks → Create Task → Add labels manually
```

---

# ## 9. **Upload images into the CVAT task**

Select the same images used for JSON generation.

```text
CVAT → Task → Data → Select "test/images" folder → Submit
```

---

# ## 10. **Import COCO JSON annotations into CVAT**

Load all polygon segmentation results into the task.

```text
CVAT → Task → Actions → Upload annotations → Format: COCO 1.0 → Select coco_sam2_for_cvat.json
```

---

# ## 11. **Visually inspect and refine polygons in CVAT**

Use the polygon editing tool to adjust vertices and correct segmentation errors.

```text
CVAT editor → Polygon tool → Edit → Save
```

---

# ## 12. **Export refined annotations (optional)**

Export the cleaned segmentation dataset for YOLOv8-seg training.

```text
CVAT → Task → Actions → Export annotations → YOLO Segmentation 1.0
```

---

# ## 13. **Train YOLOv8-seg with your refined segmentation dataset**

Use the exported YOLO-seg labels to train the segmentation model.

```bash
yolo train model=yolov8s-seg.pt data=your_data.yaml imgsz=640 epochs=100
```

---

# ## ✔️ Final Summary

You have now built a **full industrial-grade annotation pipeline**:

1. Activate environment
2. Start CVAT
3. Run SAM2 segmentation on YOLO boxes
4. Generate YOLO-seg polygon labels
5. Convert to COCO JSON
6. Import into CVAT
7. Refine masks
8. Export final dataset
9. Train YOLOv8-seg


