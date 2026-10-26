# HoleIO

PPO agent (Stable-Baselines3) that plays Hole.io (solo mode, Japan map) using screen capture and mouse control.

## Setup

```bash
pip install -r requirements.txt
```

## Calibration

The constants at the top of `holeIoAgent.py` have to match your screen:

- `GAME_BOX`: position and size of the game window on screen
- `SCORE_ROI`: area of the percentage display (relative to `GAME_BOX`)
- `CENTER_ROI`: area around the hole used for stuck detection (relative to `GAME_BOX`)
- `retryButton.png`: screenshot of the retry button at game scale (optional; without it the button is not detected)
- `digits/0.png` ... `digits/9.png`: digit templates. Synthetic ones are generated if missing; crops of the real in-game digits give better OCR results.

## Training

```bash
python holeIoAgent.py
```

After starting you have 3 seconds to focus the game window. Ctrl+C stops training and still saves `holeIoPpoModel.zip`. Each episode is logged to `runsData.csv`.
