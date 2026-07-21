# Port Container Intelligence

Port Container Intelligence is a Python-based computer vision pipeline for analyzing port and harbor video footage. It identifies and tracks container cranes and ships, overlays segmentation masks on the video, and generates a summary dashboard with detection trends and activity heatmaps.

The project is designed for visual inspection of port operations, including ship docking activity, crane movement, and overall operational activity over time.

## What the project does

The script processes an input video and produces:

- An annotated output video with:
  - colored segmentation overlays for ships and cranes
  - bounding boxes and tracking labels
  - movement trails and a compact HUD panel
  - water masking so water regions are not over-colored
- A dashboard image summarizing:
  - detections over time
  - object count distribution
  - detection split between cranes and ships
  - activity heatmap

## Key features

- Detects and tracks ships and cranes in video footage
- Uses color-based segmentation and background subtraction
- Excludes water regions from segmentation overlays
- Applies lightweight object tracking with persistence across frames
- Generates a visual analytics dashboard
- Automatically installs missing Python dependencies when needed

## Project structure

- `port-container-intelligence.py` – main processing script
- `port_output_v6/` – output directory for generated video and dashboard files
- `port_output_v6.mp4` / `port_output_v6.avi` – rendered annotated video output
- `port_dashboard_v6.png` – generated dashboard image

## Requirements

The script uses the following Python packages:

- `opencv-python`
- `numpy`
- `matplotlib`
- `tqdm`

If any are missing, the script will try to install them automatically.

### Python version

A recent Python 3.x installation is recommended (Python 3.8+ is a good target).

## How to run

Run the script from the project root:

```bash
python port-container-intelligence.py
```

If you provide a video path as an argument, the script will use that file:

```bash
python port-container-intelligence.py /path/to/your/video.mp4
```

If no argument is provided, the script searches the current working directory for a supported video file matching common patterns such as:

- `*.mp4`
- `*.avi`
- `*.mov`
- `*.mkv`

## Output files

After processing completes, the script writes:

- `port_output_v6/port_output_v6.mp4` or `port_output_v6/port_output_v6.avi`
- `port_output_v6/port_dashboard_v6.png`
- and copies the generated outputs to the project root as:
  - `port_output_v6.avi` or `port_output_v6.mp4`
  - `port_dashboard_v6.png`

## Notes

- The processing is optimized for port/harbor-like scenes and may require tuning for different environments.
- Detection performance depends heavily on video quality, camera angle, lighting, and scene complexity.
- The script uses a conservative segmentation and tracking approach to avoid over-detection.

## Example workflow

1. Place a video file in the project directory or pass its path to the script.
2. Run the analysis command.
3. Open the generated video and dashboard outputs for review.

## License

This project is provided as-is for research, experimentation, and visualization purposes.
