This is a deepfake filter applied on a live webcam frame or on a sample video, which recognizes various faces through Haar_Cascades detection system of openCV and uses dlib's face_landmark_68_detector which detects various landmarks such as eyes, nose, ears, etc. in a face and then with the help of openCV we make a mask of that celeb_image & warp them accordingly & blends them onto our frames.

Test Video : "https://drive.google.com/file/d/1TLkp1uSyE1lulBe9NCM12O_ZVgOT80bx/view?usp=sharing"
