import csv
import math
import os
import time

import cv2
import gymnasium
import mss
import numpy as np
import pyautogui
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Calibration constants (ROIs are relative to GAME_BOX)
GAME_BOX = {"top": 100, "left": 100, "width": 800, "height": 600}
SCORE_ROI = {"top": 15, "left": 330, "width": 140, "height": 45}
CENTER_ROI = {"top": 200, "left": 300, "width": 200, "height": 200}
RETRY_TEMPLATE_PATH = "retryButton.png"
DIGIT_TEMPLATES_DIR = "digits/"

# Runtime configuration
runsCsvPath = "runsData.csv"
modelSavePath = "holeIoPpoModel.zip"
observationSize = 84
actionCount = 8
actionRadius = 150
maxEpisodeSeconds = 120.0
maxStuckSteps = 30
stuckDiffThreshold = 2.0
retryConfidence = 0.8
digitTemplateWidth = 20
digitTemplateHeight = 32
digitMatchThreshold = 0.5
maxDigitAspectRatio = 0.9
maxPercentageJump = 15.0
jumpConfirmFrames = 3
moveDurationSeconds = 0.04
moveSubsteps = 4
stepDelaySeconds = 0.05
retryWaitSeconds = 10.0
resetTimeoutSeconds = 20.0
retryClickInterval = 2.0
pollIntervalSeconds = 0.25
startupDelaySeconds = 3.0

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


# Crops a {"top","left","width","height"} region from an image
def cropRoi(image, roi):
    return image[roi["top"]:roi["top"] + roi["height"], roi["left"]:roi["left"] + roi["width"]]


# Otsu threshold with foreground forced to white (border pixels are treated as background)
def binarizeImage(grayImage):
    _, binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borderPixels = np.concatenate([binaryImage[0, :], binaryImage[-1, :], binaryImage[:, 0], binaryImage[:, -1]])
    if np.mean(borderPixels) > 127:
        binaryImage = cv2.bitwise_not(binaryImage)
    return binaryImage


# Crops a glyph to its bounding box and fits it centered into the template size, keeping aspect ratio
def normalizeGlyph(binaryGlyph):
    points = cv2.findNonZero(binaryGlyph)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    glyph = binaryGlyph[y:y + h, x:x + w]
    scale = min(digitTemplateWidth / w, digitTemplateHeight / h)
    newWidth = min(digitTemplateWidth, max(1, int(round(w * scale))))
    newHeight = min(digitTemplateHeight, max(1, int(round(h * scale))))
    resized = cv2.resize(glyph, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((digitTemplateHeight, digitTemplateWidth), dtype=np.uint8)
    offsetX = (digitTemplateWidth - newWidth) // 2
    offsetY = (digitTemplateHeight - newHeight) // 2
    canvas[offsetY:offsetY + newHeight, offsetX:offsetX + newWidth] = resized
    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
    return canvas


# Renders missing 0-9 digit templates with OpenCV fonts
def generateSyntheticDigitTemplates(directory):
    os.makedirs(directory, exist_ok=True)
    for digit in range(10):
        templatePath = os.path.join(directory, f"{digit}.png")
        if os.path.isfile(templatePath):
            continue
        canvas = np.zeros((100, 80), dtype=np.uint8)
        cv2.putText(canvas, str(digit), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 7, cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(templatePath, normalizeGlyph(canvas))


# Loads digit templates as normalized float32 glyphs, generating missing files first
def loadDigitTemplates(directory):
    generateSyntheticDigitTemplates(directory)
    digitTemplates = {}
    for digit in range(10):
        templateImage = cv2.imread(os.path.join(directory, f"{digit}.png"), cv2.IMREAD_GRAYSCALE)
        if templateImage is None:
            raise RuntimeError(f"Unable to read digit template {digit} in {directory}")
        glyph = normalizeGlyph(binarizeImage(templateImage))
        if glyph is None:
            raise RuntimeError(f"Digit template {digit} in {directory} is empty")
        digitTemplates[digit] = glyph.astype(np.float32)
    return digitTemplates


# Loads the retry button template as grayscale, or None if missing
def loadRetryTemplate(templatePath):
    if not os.path.isfile(templatePath):
        print(f"Warning: {templatePath} not found, retry button detection disabled.")
        return None
    return cv2.imread(templatePath, cv2.IMREAD_GRAYSCALE)


# Merges bounding boxes that overlap horizontally (e.g. the parts of a "%" sign)
def mergeOverlappingBoxes(boxes):
    sortedBoxes = sorted(boxes, key=lambda box: box[0])
    mergedBoxes = [list(sortedBoxes[0])]
    for x, y, w, h in sortedBoxes[1:]:
        lastBox = mergedBoxes[-1]
        if x < lastBox[0] + lastBox[2] - 1:
            right = max(lastBox[0] + lastBox[2], x + w)
            bottom = max(lastBox[1] + lastBox[3], y + h)
            lastBox[1] = min(lastBox[1], y)
            lastBox[2] = right - lastBox[0]
            lastBox[3] = bottom - lastBox[1]
        else:
            mergedBoxes.append([x, y, w, h])
    return [tuple(box) for box in mergedBoxes]


# Returns the best matching digit and its normalized correlation score
def matchDigit(glyph, digitTemplates):
    glyphFloat = glyph.astype(np.float32)
    bestDigit, bestScore = None, -1.0
    for digit, template in digitTemplates.items():
        score = float(cv2.matchTemplate(glyphFloat, template, cv2.TM_CCOEFF_NORMED)[0][0])
        if not np.isnan(score) and score > bestScore:
            bestDigit, bestScore = digit, score
    return bestDigit, bestScore


# Contour-based OCR: segments glyphs left to right and template-matches digits and a decimal point
def readDigits(binaryImage, digitTemplates):
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    boxes = [box for box in boxes if box[2] * box[3] >= 4]
    if not boxes:
        return None
    boxes = mergeOverlappingBoxes(boxes)
    maxHeight = max(box[3] for box in boxes)
    if maxHeight < binaryImage.shape[0] * 0.3:
        return None
    baseline = max(box[1] + box[3] for box in boxes if box[3] >= 0.6 * maxHeight)
    text = ""
    for x, y, w, h in boxes:
        if h >= 0.6 * maxHeight:
            glyph = normalizeGlyph(binaryImage[y:y + h, x:x + w])
            if glyph is None:
                continue
            digit, score = matchDigit(glyph, digitTemplates)
            if w / h > maxDigitAspectRatio or score < digitMatchThreshold:
                if text:
                    break
                continue
            text += str(digit)
        elif h <= 0.35 * maxHeight and abs((y + h) - baseline) <= 0.2 * maxHeight and text and "." not in text:
            text += "."
    text = text.rstrip(".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value <= 100.0 else None


# Returns the number of episodes already logged in the CSV file
def countLoggedEpisodes(csvPath):
    if not os.path.isfile(csvPath):
        return 0
    try:
        with open(csvPath, newline="") as csvFile:
            rows = list(csv.reader(csvFile))
        return max(0, len(rows) - 1)
    except OSError:
        return 0


class HoleIoEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(observationSize, observationSize, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(actionCount)
        self.screenCapture = mss.mss()
        self.digitTemplates = loadDigitTemplates(DIGIT_TEMPLATES_DIR)
        self.retryTemplate = loadRetryTemplate(RETRY_TEMPLATE_PATH)
        self.centerX = GAME_BOX["left"] + GAME_BOX["width"] // 2
        self.centerY = GAME_BOX["top"] + GAME_BOX["height"] // 2
        self.actionOffsets = [
            (int(round(actionRadius * math.cos(index * math.pi / 4))), int(round(-actionRadius * math.sin(index * math.pi / 4))))
            for index in range(actionCount)
        ]
        self.maxStuckSteps = maxStuckSteps
        self.isMouseDown = False
        self.terminationReason = None
        self.episodeNumber = countLoggedEpisodes(runsCsvPath)
        self.resetEpisodeMetrics(None)

    # Resets all per-episode tracking values
    def resetEpisodeMetrics(self, frame):
        self.episodeStartTime = time.time()
        self.lastStepTime = self.episodeStartTime
        self.episodeSteps = 0
        self.currentPercentage = 0.0
        self.previousPercentage = 0.0
        self.maxPercentage = 0.0
        self.totalEpisodeReward = 0.0
        self.unchangedSteps = 0
        self.pendingPercentage = None
        self.pendingConfirmations = 0
        self.previousFrame = frame
        self.terminationReason = None

    # Grabs GAME_BOX and converts BGRA to grayscale
    def captureFrame(self):
        rawFrame = np.array(self.screenCapture.grab(GAME_BOX))
        return cv2.cvtColor(rawFrame, cv2.COLOR_BGRA2GRAY)

    # Resizes a grayscale frame to the 84x84x1 observation
    def preprocessObservation(self, frame):
        resized = cv2.resize(frame, (observationSize, observationSize), interpolation=cv2.INTER_AREA)
        return resized.reshape(observationSize, observationSize, 1).astype(np.uint8)

    # Reads the raw percentage from SCORE_ROI, or None if unreadable
    def readRawPercentage(self, frame):
        binaryRoi = binarizeImage(cropRoi(frame, SCORE_ROI))
        return readDigits(binaryRoi, self.digitTemplates)

    # Reads the percentage and filters out decreases and unconfirmed large jumps
    def extractPercentage(self, frame):
        rawValue = self.readRawPercentage(frame)
        if rawValue is None:
            return self.currentPercentage
        rawValue = float(np.clip(rawValue, 0.0, 100.0))
        if rawValue < self.currentPercentage:
            return self.currentPercentage
        if rawValue - self.currentPercentage > maxPercentageJump:
            if self.pendingPercentage is not None and abs(rawValue - self.pendingPercentage) < 0.5:
                self.pendingConfirmations += 1
            else:
                self.pendingPercentage = rawValue
                self.pendingConfirmations = 1
            if self.pendingConfirmations < jumpConfirmFrames:
                return self.currentPercentage
        self.pendingPercentage = None
        self.pendingConfirmations = 0
        return rawValue

    # Detects stuck state via low center-region motion or a stalled percentage
    def checkIfStuck(self, currentFrame, previousFrame):
        isVisuallyStuck = False
        if previousFrame is not None:
            difference = cv2.absdiff(cropRoi(currentFrame, CENTER_ROI), cropRoi(previousFrame, CENTER_ROI))
            isVisuallyStuck = float(np.mean(difference)) < stuckDiffThreshold
        if self.currentPercentage == self.previousPercentage:
            self.unchangedSteps += 1
        else:
            self.unchangedSteps = 0
        return isVisuallyStuck or self.unchangedSteps >= self.maxStuckSteps

    # Locates the retry button and returns (confidence, screen position)
    def findRetryButton(self, frame):
        if self.retryTemplate is None:
            return 0.0, None
        templateHeight, templateWidth = self.retryTemplate.shape[:2]
        if templateHeight > frame.shape[0] or templateWidth > frame.shape[1]:
            return 0.0, None
        result = cv2.matchTemplate(frame, self.retryTemplate, cv2.TM_CCOEFF_NORMED)
        _, maxValue, _, maxLocation = cv2.minMaxLoc(result)
        screenPosition = (
            GAME_BOX["left"] + maxLocation[0] + templateWidth // 2,
            GAME_BOX["top"] + maxLocation[1] + templateHeight // 2,
        )
        return float(maxValue), screenPosition

    # Ends the episode on 100%, timeout, or a visible retry button
    def checkTermination(self, frame, currentPercentage, elapsedSeconds):
        if currentPercentage >= 100.0:
            self.terminationReason = "complete"
        elif elapsedSeconds >= maxEpisodeSeconds:
            self.terminationReason = "timeout"
        elif self.findRetryButton(frame)[0] >= retryConfidence:
            self.terminationReason = "retry"
        else:
            return False
        return True

    def calculateReward(self, currentPercentage, previousPercentage, isStuck):
        if currentPercentage >= 100.0:
            return 1000.0
        if currentPercentage > previousPercentage:
            return (currentPercentage - previousPercentage) * 10.0
        if currentPercentage == previousPercentage or isStuck:
            return -1.0
        return 0.0

    # Holds the left mouse button at the screen center if not already held
    def pressMouse(self):
        if not self.isMouseDown:
            pyautogui.moveTo(self.centerX, self.centerY)
            pyautogui.mouseDown()
            self.isMouseDown = True

    def releaseMouse(self):
        pyautogui.mouseUp()
        self.isMouseDown = False

    # Interpolates the cursor towards the target in small substeps
    def moveCursorSmoothly(self, targetX, targetY):
        startX, startY = pyautogui.position()
        for substep in range(1, moveSubsteps + 1):
            progress = substep / moveSubsteps
            pyautogui.moveTo(int(startX + (targetX - startX) * progress), int(startY + (targetY - startY) * progress))
            time.sleep(moveDurationSeconds / moveSubsteps)

    # Clicks the retry button if visible and returns whether it was clicked
    def clickRetryButton(self, frame):
        confidence, position = self.findRetryButton(frame)
        if confidence >= retryConfidence and position is not None:
            pyautogui.click(position[0], position[1])
            return True
        return False

    # Polls for the retry button after an episode and clicks it once found
    def waitForRetryButton(self):
        deadline = time.time() + retryWaitSeconds
        while time.time() < deadline:
            if self.clickRetryButton(self.captureFrame()):
                return True
            time.sleep(pollIntervalSeconds)
        return False

    # Polls until the retry button is gone and the percentage HUD is readable
    def waitForGameActive(self):
        deadline = time.time() + resetTimeoutSeconds
        lastClickTime = time.time()
        while time.time() < deadline:
            frame = self.captureFrame()
            if self.findRetryButton(frame)[0] >= retryConfidence:
                if time.time() - lastClickTime >= retryClickInterval and self.clickRetryButton(frame):
                    lastClickTime = time.time()
            elif self.readRawPercentage(frame) is not None:
                return frame
            time.sleep(pollIntervalSeconds)
        print("Warning: game HUD not detected before timeout, continuing anyway.")
        return self.captureFrame()

    # Appends the finished episode's stats to the CSV log
    def logEpisode(self):
        durationSeconds = self.lastStepTime - self.episodeStartTime
        fileExists = os.path.isfile(runsCsvPath)
        with open(runsCsvPath, "a", newline="") as csvFile:
            writer = csv.writer(csvFile)
            if not fileExists:
                writer.writerow(["episodeNumber", "maxPercentage", "durationSeconds", "totalEpisodeReward"])
            writer.writerow([self.episodeNumber, self.maxPercentage, round(durationSeconds, 2), round(self.totalEpisodeReward, 2)])
        print(
            f"Episode {self.episodeNumber}: max {self.maxPercentage:.1f}% in {durationSeconds:.1f}s, "
            f"reward {self.totalEpisodeReward:.1f} ({self.terminationReason})"
        )
        self.episodeSteps = 0

    def step(self, action):
        self.pressMouse()
        offsetX, offsetY = self.actionOffsets[int(action)]
        self.moveCursorSmoothly(self.centerX + offsetX, self.centerY + offsetY)
        time.sleep(stepDelaySeconds)

        frame = self.captureFrame()
        self.lastStepTime = time.time()
        self.previousPercentage = self.currentPercentage
        self.currentPercentage = self.extractPercentage(frame)
        self.maxPercentage = max(self.maxPercentage, self.currentPercentage)
        elapsedSeconds = self.lastStepTime - self.episodeStartTime

        isStuck = self.checkIfStuck(frame, self.previousFrame)
        terminated = self.checkTermination(frame, self.currentPercentage, elapsedSeconds)
        reward = self.calculateReward(self.currentPercentage, self.previousPercentage, isStuck)

        self.totalEpisodeReward += reward
        self.episodeSteps += 1
        self.previousFrame = frame
        if terminated:
            self.releaseMouse()

        info = {
            "percentage": self.currentPercentage,
            "isStuck": isStuck,
            "elapsedSeconds": elapsedSeconds,
            "terminationReason": self.terminationReason,
        }
        return self.preprocessObservation(frame), reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.releaseMouse()
        if self.episodeSteps > 0:
            self.logEpisode()
            self.waitForRetryButton()
        else:
            self.clickRetryButton(self.captureFrame())
        frame = self.waitForGameActive()

        self.episodeNumber += 1
        self.resetEpisodeMetrics(frame)
        initialPercentage = self.readRawPercentage(frame)
        if initialPercentage is not None:
            self.currentPercentage = self.previousPercentage = self.maxPercentage = float(np.clip(initialPercentage, 0.0, 100.0))
        return self.preprocessObservation(frame), {}

    def close(self):
        if self.episodeSteps > 0:
            self.logEpisode()
        self.releaseMouse()
        if self.screenCapture is not None:
            self.screenCapture.close()
            self.screenCapture = None


if __name__ == "__main__":
    print(f"Starting in {startupDelaySeconds:.0f} seconds, focus the Hole.io window (Solo, Japan)...")
    time.sleep(startupDelaySeconds)

    env = VecFrameStack(DummyVecEnv([lambda: HoleIoEnv()]), n_stack=4)
    model = PPO("CnnPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)
    try:
        model.learn(total_timesteps=100_000)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        pyautogui.mouseUp()
        model.save(modelSavePath)
        env.close()
        print(f"Model saved to {modelSavePath}")
