# AI Camera

Self-hosted AI CCTV dashboard: live cameras, object detection and tracking, zones, line counting,
pose and fall detection, face blur, heatmap, motion, alerts with photos and clips, Telegram, analytics,
and quality inspection (anomaly detection, e.g. egg lines). Monitoring and local training run on your device.
Roboflow upload/download and cloud training are optional, explicit actions.
Includes camera-outage/recovery alerts, per-camera alert schedules, activity confirmation, a look-based
gate open/closed sensor, opt-in recording retention, and configuration backup/restore. Automated checks
gate container releases.

- **Hailo-8 / 8L** (Raspberry Pi 5, CM5) runs detection, pose and faces on the AI chip.
- **Any PC without Hailo** falls back to CPU (Ultralytics YOLO), slower.

![Live overview dashboard](docs/live-overview.png)

## Screenshot guide

**Live overview** — all cameras, KPI tiles (cameras online, objects in view, people/vehicles today, alerts),
the camera-feed grid with per-feed Expand / Snapshot / Record / Pause / Freeze, and a Recent activity feed.

![Live overview](docs/live-overview.png)

**First run** — before any camera is added you get an empty state. Click **Set up a camera** (or **Manage cameras**)
to open Settings → Cameras.

<img src="docs/first-run.png" alt="First-run empty state" width="320">

**AI Training → Photos & labels** — capture or upload frames, split them into training/validation sessions,
draw boxes, and mark each photo reviewed. Only reviewed labels enter local training.

![AI Training photos and labels](docs/ai-training.png)

**AI Training → Models & evaluation** — compare candidate detectors on the same validation set
(precision, recall, mAP@50, mAP@50–95), then activate one on the cameras you choose.

![Model comparison](docs/model-comparison.png)

**On a phone** — the whole dashboard is responsive and works in a mobile browser.

<img src="docs/live-overview-mobile.png" alt="Live overview on a phone" width="300">

## Run on a PC (Windows / Linux)

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux: .venv/bin/pip
.venv/Scripts/python app.py
```

Open http://127.0.0.1:8080 and log in with `admin` / `admin` (change it in Settings → Login).
Add cameras in Settings → Cameras (RTSP brands, USB webcam, HTTP stream, video file; **Find cameras** scans the network).

## Run with Docker

A prebuilt multi-arch image (amd64 + arm64) is published to GitHub Packages after automated checks
pass on pushes to `main`, version tags, and manual workflow runs:

```
docker run -p 8080:8080 \
  -v ai-camera-data:/app/data \
  -v ai-camera-events:/app/events -v ai-camera-workspace:/app/ai-workspace \
  ghcr.io/dineshpalsingh1989/ai-camera:latest
```

This runs the dashboard with CPU detection. Hailo acceleration needs HailoRT and `/dev/hailo0`
passed through from the host, so for a Raspberry Pi + Hailo install, use `install.sh` below instead.

The `ai-camera-data` volume preserves login/camera settings, statistics, inspection models and
collected photos when replacing the container. Events and the reviewed training workspace retain
their existing volumes. Before upgrading an older container, stop it and copy `/app/settings.json`,
`/app/stats.json`, `/app/inspect/` and `/app/training/` (when present) into the new data volume,
preserving these names. Keep the old container until the restored configuration has been verified.
PC/Pi installs keep their current data locations unless `AI_CAMERA_DATA_DIR` is explicitly set;
this variable relocates those four items only, not events or the reviewed training workspace.

## Install on a Raspberry Pi with Hailo

```
bash install.sh
```

Installs dependencies, registers the `ai-camera` service (starts on boot) and runs the self-check.
Quality inspection additionally needs PyTorch (`pip install torch torchvision`).

## Demo videos

`python make_demo_eggs.py` creates synthetic egg-line videos in `demo-videos/` for trying inspection without a real line.

## Custom objects (e.g. egg)

Settings → Detection → Custom objects. `yolov8s-worldv2.pt` finds objects by name with no training.
Use **AI Training** to collect, label, train, evaluate and activate a compatible Ultralytics object detector.
Custom detectors currently run on CPU. Downloading a `.pt` model does not compile it for Hailo; Hailo's
standard detection/pose/face models still use their configured `.hef` files.

## Monitoring workspace

- Set a **camera group** in Settings → Cameras. Search cameras by name, location or group on Live overview.
- **Focus** enlarges a feed while keeping the remaining feeds alongside it. Exit focus restores the grid.
- **Camera health** shows frame timestamps, connection times, AI FPS and reconnection attempts since startup.
- Filter **Event history** by dates (browser-local time), severity and review status. Open an event to mark
  it confirmed, a false alarm or unreviewed, with a note. These reviews do not automatically label training images.
- Settings are organized into Cameras, Detection, Recording, Notifications and Login tabs.

Face privacy remains active when AI is paused. If face detection is unavailable or fails while
privacy is enabled, the live image is hidden. Manual **Record** works even when automatic alert
clips are disabled; encoder and low-storage failures are reported instead of showing success.

## Unattended monitoring

Save settings after changing any of these controls. Existing installations receive the new defaults
on their next server start. Cleanup is disabled by default and does not remove existing files until enabled.

| Feature | Settings location | Default |
| --- | --- | --- |
| Camera offline/recovery alerts | Notifications → Camera health | Enabled; 60-second outage delay |
| Activity confirmation | Notifications → Reduce repeated alerts | 2 seconds |
| Inspection alert grouping | Notifications → Reduce repeated alerts | Enabled; uses Detection's 60-second alert cooldown |
| Alert schedule | Cameras → select a camera → Alert schedule | Disabled (alerts always active) |
| Gate sensor (open/closed by look) | Cameras → select a camera → Zones, lines & gates | Off until a gate is drawn and calibrated |
| Recording cleanup | Recording → Storage retention | Disabled; 30 days when enabled |
| Free-space reserve | Recording → Storage retention | 1 GiB |
| Configuration backup/restore | Login → Configuration backup | Manual download and restore |

### Camera offline and recovery alerts

A background check runs every five seconds. It detects disconnected cameras and streams whose
last frame is at least 15 seconds old, even if the capture connection still claims to be online.
After the configured outage delay, one **Camera offline** event is recorded. Startup has the same
grace period so a camera has time to connect. A **Camera recovered** event follows after five seconds
of healthy checks. Brief recoveries do not repeatedly notify you. Disabled/removed cameras are not checked.

Health events appear in Event history, including its new type filters, and send Telegram text messages
when a bot token and chat ID are saved. They do not create video clips or trigger the hardware siren.
They remain enabled outside detection schedules and while AI is paused. Timers reset when the server restarts.

### Per-camera alert schedules

Choose weekdays, a start/end time and an IANA time zone such as `Asia/Singapore`.
For example, select Monday–Friday and **20:00–06:00** to monitor overnight: Friday's interval
continues until Saturday at 06:00. The end time is exclusive. Equal start/end times mean the full
selected calendar day. Time-zone rules include daylight-saving changes; schedules do not depend
on the server's operating-system time zone. Check the time-zone field when first enabling a schedule;
older camera configurations start with UTC, while newly added cameras use the browser's time zone.

Outside scheduled hours, automatic detection events, their clips, Telegram notifications and siren
actions are suppressed. Live video, privacy blur, object/line counting, training collection, manual
recordings and camera-health alerts remain available. Live tiles show **ALERTS OUTSIDE SCHEDULE**.
An event clip already recording is allowed to finish.

### Smarter alerts

**Activity confirmation** waits for tracked objects, zone entries and motion to persist before generating
events. Set it to **0** for immediate alerts. Object tracking tolerates short detection gaps; a gap longer
than five seconds starts a new track. This is a persistence filter, not a guarantee that every detection
is correct. Existing fall confirmation, zone dwell times and inspection voting still apply independently.

The existing cooldown suppresses repeated camera/type/label events. **Inspection alert grouping** also
ignores changing tracked object IDs and defect scores, so consecutive rejected items of the same class
share a cooldown instead of generating a notification for each new ID. Turn grouping off when individual
reject alerts matter. Manual recordings and health transitions bypass this cooldown.

### Storage retention

Enable **Automatically delete old event photos and clips**, choose retention days, and set the disk reserve.
Cleanup runs every minute. It removes expired events, then the oldest remaining events if free space is
below the reserve. Set days to **0** to use free-space cleanup alone. Active clips and events created in
the last two minutes are protected. App-named orphan media and abandoned temporary clips older than a day
are also removed. Cleanup never removes training images, datasets, model files or unrelated media files.

The Recording pane displays free space, the last maintenance time, removed-event count and errors.
The event-history cap remains approximately 500 entries, independent of the retention setting.
The reserve also prevents new recordings from starting when insufficient space is available, even with
cleanup disabled. It cannot guarantee free space if other applications fill the disk, or if only protected
recordings remain. A manual recording reports this condition as an error.

### Gate sensor (open/closed by look)

A gate watches a drawn rectangle for a change in appearance instead of tracking objects — useful for a
gate, door or barrier that a person-detector can't reliably see as "open" or "closed". In a camera's
**Zones, lines & gates** editor, click **Draw gate** and click two opposite corners around the gate
panel itself (avoid sky, moving trees or reflective surfaces in the box). Save, then with the gate
physically **closed**, click that gate's **Calibrate (closed now)** button and save again.

Every frame, the current look of the box is compared to its calibrated reference (a small blurred,
downsampled grayscale patch, matched by normalized correlation); a sustained mismatch, held for the
activity-confirmation setting above, means **open**. The match is smoothed over time to ignore ordinary
camera noise (compression, auto-exposure flicker) rather than reacting to a single frame. The live video
draws the box directly on the feed, labelled OPEN (red) or CLOSED (green), alongside a live status next
to each gate in its editor row (uncalibrated / calibrating / open / closed).

Gate open/closed transitions raise a normal `gate` event — Telegram, siren, Event history, alert
cooldown and the alert schedule all apply exactly as they do for object/intrusion alerts. This is a
visual look-alike check, not true object understanding: a strong lighting change across the exact drawn
box (sun, headlights) can occasionally misread as "open" even if nothing moved. Recalibrate after
repositioning a camera, or raise the match threshold (and tighten the box) if you see false opens. A
physical sensor (e.g. a magnetic reed switch wired to one of the IRIV board's digital inputs, shown
on the in-app Hardware page) stays the more reliable option where that matters.

## Configuration backup and restore

1. Open **Settings → Login → Configuration backup** and download the saved configuration.
   Save pending settings edits first; the download contains the server's saved configuration.
2. Keep the JSON file private: it contains camera connection credentials, dashboard login and the Telegram token.
3. To restore, select the JSON file. The server validates its version, cameras, schedules and collection rules
   and shows the number of cameras/rules before any settings are changed.
4. Select **Restore configuration** and confirm replacement. Running training/transfer jobs must finish first.
   Log in using the restored credentials. Restart the application to reload base AI model paths.

Backups include global settings, camera connections, zones/lines, schedules, inspection settings,
custom-model path/class configuration, and per-camera automatic training collection rules.
They exclude recordings, statistics, learned inspection model files, training photos, datasets, model
weights and the Roboflow connection/API key. Existing training artifacts and Roboflow credentials remain
on the target installation; activation rollback history is cleared because the configuration was replaced.
Use the existing dataset/model downloads and filesystem backups for those artifacts. Model paths in a
configuration backup must exist on the target machine; restore does not download or import weights.
Backups use version 1 of the `ai-camera-configuration` JSON format and are limited to 5 MB.

Downloads require dashboard authentication and are not cached. Preview is read-only; restore requires
explicit confirmation. Failed validation never applies settings, and handled storage-write failures
roll back the training collection configuration before reporting an error.

## Reviewed training workflow

Install training dependencies into the same virtual environment:

```
.venv/Scripts/python -m pip install -r requirements-training.txt
```

1. In **AI Training → Photos & labels**, capture frames, upload JPEG/PNG images, or import the latest 500
   photos from the existing per-camera collection tool. Optional uncertain-detection collection is off by default,
   samples changing scenes at a bounded interval, and never marks its own guesses as reviewed.
2. Open each photo, draw or enter object boxes, assign classes, and check the review checkbox. An image with
   no target objects can be reviewed with zero boxes. Corrections replace suggested boxes.
3. Choose training, validation or test **capture sessions**. Use independently captured scenes to avoid near-duplicate
   frames leaking across splits. At least two training and two validation images are required; this is only a functional
   minimum, not enough for a production-quality model. Exact cross-split duplicates are rejected.
4. Build a dataset snapshot, or import a YOLO detection ZIP containing `data.yaml`, image folders and labels.
   Imports support `train/images`, `valid/images` (or `val/images`), optional `test/images`, and `images/train` layouts.
   Classification, segmentation and pose datasets are not accepted. ZIP import is capped at 1 GB compressed / 4 GB unpacked.
5. Start local training from the installed `yolov8n.pt` or a model in the library. CPU is the default; GPU 0 requires
   a working CUDA PyTorch installation. Training runs in a separate cancellable process; monitoring may run slower.
6. Use **Add current model for comparison** to copy the existing custom detector into the model library.
   Compare candidates on **the same dataset** using precision, recall, mAP@50 and mAP@50–95. Imported models must
   be evaluated before activation. The dataset class names and order must match the candidate model.
7. Select cameras and activate the candidate. Activation updates the shared custom detector and enables it on those
   cameras only. The previous model, class names and per-camera custom-detection selection are retained for rollback.

Dataset snapshots are immutable; editing/deleting source photos does not change an existing snapshot. Both datasets
and model weights can be downloaded from their library. Jobs interrupted by a server restart are marked interrupted;
they do not silently restart or activate models. Only import model weights from sources you trust.

## Roboflow

Enter a private API key, workspace slug and project slug under **AI Training → Roboflow connection**.
The key stays in ignored server-side `ai-workspace/connection.json`; API responses never include it.
No account access is needed for the local workflow. There is no automatic cloud synchronization.

- Select photos and choose **Upload selected to Roboflow**. Reviewed boxes upload as annotations; unreviewed
  suggestions do not. Successful uploads are tracked per destination and annotation revision, so retrying skips
  unchanged photos already uploaded. Use the project link to finish labeling and generate a version on Roboflow.
- **Download YOLO dataset** imports the selected dataset version for local training.
- **Download model weights** accepts a model slug or version number and checks compatibility. A `.pt` extension
  alone does not imply compatibility: this application supports Ultralytics object detection, not RF-DETR or other runtimes.
- **Submit cloud training** explicitly starts a job using the model type identifier supported by your Roboflow project
  (leave it blank to use Roboflow's default).
  This can consume account credits. Cloud jobs are monitored using **Refresh cloud status**; manage cloud cancellation
  from Roboflow. If submission times out, check Roboflow before submitting again to avoid duplicate paid jobs.

Supported exports and model identifiers depend on Roboflow and the project's architecture. Integration endpoints follow
the [official Roboflow API adapter](https://github.com/roboflow/roboflow-python/blob/main/roboflow/adapters/rfapi.py).

## Self-check

```
python test_app.py
python -m unittest test_workspace -v
python -m unittest test_runtime -v
python -m unittest test_monitoring -v
python check_frontend.py  # requires Node.js
```

The self-check uses a disposable app copy. API/policy tests use temporary directories and simulated
cameras; they do not change a running installation or send cloud requests. Tests cover offline/recovery
transitions, overnight/DST schedules, persistence/cooldown rules, safe cleanup, authentication,
privacy behavior, and backup/restore validation and write failures.

For a browser smoke check, install `requirements-test.txt` in a separate test environment, then run:

```
python -m playwright install chromium
python check_dashboard.py
```

It launches an isolated local server and checks saving settings, schedules, alerts, retention,
backup download/preview/restore, and mobile controls. `BROWSER_EXECUTABLE` can point to an installed
Chromium/Edge executable instead. The lightweight test dependencies omit PyTorch/Ultralytics;
real inference/training and hardware checks remain separate.

### Release checks

`.github/workflows/docker.yml` runs the self-check, runtime/workspace/monitoring tests, JavaScript syntax
checks and the browser smoke check on pull requests and release triggers. The container publishing job
depends on successful checks; pull requests never publish an image. Package-write permission is limited
to the publishing job. CI uses simulated cameras and does not validate a physical Hailo board, RTSP camera,
Telegram delivery or a production video encoder. The optional real-training check below is not part of CI.

Optional real training smoke check (one epoch on synthetic fixture images, never deployed):

```
# PowerShell
$env:RUN_TRAINING_SMOKE='1'
.venv/Scripts/python -m unittest test_workspace.WorkspaceTests.test_real_one_epoch_training_and_validation -v
```

`settings.json` (passwords), events, learned models, training photos, and `ai-workspace/` (datasets, labels,
credentials, job logs, registry and model versions) are ignored by Git. Back up these folders to retain learning history.
