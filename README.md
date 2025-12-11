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

