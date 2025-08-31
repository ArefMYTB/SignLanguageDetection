import cv2
import torch
import numpy as np
import imageio

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(input_dim=1662, num_classes=len(actions)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ====== Helper functions ======
def extract_keypoints(frame):
    """
    Replace this function with your actual keypoint extractor (e.g., Mediapipe).
    Currently returns random features for demo purposes.
    """
    return np.random.rand(1662).astype(np.float32)

def predict_sequence(frames, window_size=30):
    """
    Collects keypoints from frames and runs them through the model.
    """
    sequence = [extract_keypoints(f) for f in frames]
    sequence = np.array(sequence)
    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, D]

    with torch.no_grad():
        preds = model(sequence)
        pred_class = torch.argmax(preds, dim=1).item()
    return actions[pred_class]

# ====== Main pipeline ======
def test_on_video(video_path, output_gif="result.gif", window_size=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    annotated_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) >= window_size:
            action = predict_sequence(frames[-window_size:], window_size)
            # Annotate frame
            annotated = frame.copy()
            cv2.putText(annotated, f"Prediction: {action}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        else:
            annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Save as GIF
    imageio.mimsave(output_gif, annotated_frames, fps=15, loop=0)
    print(f"Saved result gif to {output_gif}")


if __name__ == "__main__":
    test_on_video("hello.mp4", "prediction.gif")
