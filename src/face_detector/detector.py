import os
import csv
import cv2

weights_path = "./classifiers/haarcascade_frontalface_default.xml"

def detect_and_crop(path: str, out_path: str, output_size: tuple[int, int] = (350, 350),
                    nth_frame:int = 1, min_area: float = 0.05, extend: int = 50):
    """
    Detects and drops original video, to only show the most distinct face. Save into another video. 
    Args:
        path (str): Path to the video file.
        out_path (str): Path to the output video.
        output_size (tuple[int, int], optional): Returened size. Defaults to (256, 256).
        nth_frame (int): Frequency of sampling, only every nth frame will be collected.
        min_area (float, optional): Ratio of entire screen, which the face has to cover, to be detected. Defaults to 0.05.
        extend (int): Number of pixels, which extend the face bounding box in each direction. Deafults to 50.
    """


    assert os.path.exists(path)
    assert nth_frame >= 1
    assert 1 > min_area > 0
    assert type(output_size) == tuple and len(output_size) == 2 and type(output_size[0]) == type(output_size[1]) == int

    face_cascade = cv2.CascadeClassifier(weights_path)
    cap = cv2.VideoCapture(path)
    i = 0

    out_images = []

    continuous_parts = []
    start = None

    while True:
        # Load and detect face
        ret, img = cap.read()
        if not ret: break
        
        i += 1
        if i % nth_frame != 0: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.25, 3, 0)

        # Choose the most prominent one
        max_face = None
        max_area = 0
        for (x, y, w, h) in faces:
            if w * h > max_area:
                max_area = w * h
                max_face = (x - extend, y - extend, w + 2 * extend, h + 2 * extend)

        # Check face size
        if max_area <= min_area * img.shape[0] * img.shape[1]:
            max_face = None

        # Crop face
        if max_face:
            (x, y, w, h) = max_face
            x = max(0, x)
            x = min(x, img.shape[1])
            y = max(0, y)
            y = min(y, img.shape[0])
            img_i = cv2.resize(img[y: y + h, x: x + w, :], dsize=output_size)
            
            out_images.append(img_i.copy())

            if start is None: start = len(out_images) - 1
        else:
            if start is not None:
                if (len(out_images) - start) > 10:
                    continuous_parts.append((start, len(out_images)))
                start = None

    cap.release()
    cv2.destroyAllWindows()

    # Save to video
    out = cv2.VideoWriter(out_path + '.avi',  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         30, output_size)


    with open(out_path + '_segments.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(continuous_parts)


    for img in out_images:
        out.write(img)
    out.release()

if __name__ == "__main__":
    path = "./data/test4.mp4"
    path_out = "output_file"
    imgs = detect_and_crop(path, out_path=path_out, nth_frame=10)