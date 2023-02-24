import cv2
import os
""" simple video recorder, opens a cv window via webcam
    user press 'r' to start recording, the recording length will be <= 2 seconds """

class VideoRecorder:
    def __init__(self, camera = 0, name: str = None, dataPath: str = None):
        """
        Initialize the video recorder.
        :param name: gesture name for the video
        """
        self.camera = camera
        self.name = name
        self.fps = 17
        self.size = (960, 720)
        self.video = None
        self.frame_count = 0
        self.recording = False
        self.dataPath = os.path.join(os.path.abspath(os.path.join(os.path.join(dataPath, os.pardir), os.pardir)), "videos", name)

        self.save_file = os.path.join(self.dataPath, f'{self.name}.avi')
        self.cap = cv2.VideoCapture(camera)
        # cv2.resizeWindow('Gesture Recorder', self.size[0], self.size[1])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
    def run(self):
        self.start()
        while True:
            # print("recording: ", self.recording)
            _, frame = self.cap.read()
            cv2.imshow('Gesture Recorder', cv2.flip(frame, 1))
            key = cv2.waitKey(1)
            if self.recording:
                self.record(frame=frame)
            if self.handle_key(key=key):
                break

        self.cap.release()
        self.video.release()
        cv2.destroyAllWindows()

            # press 'r' to start recording press again to stop


    def handle_key(self, key: int) -> bool:
        if key == ord('r'):
            if self.recording:
                print('stop recording')
                self.stop()
                self.recording = False
                return False
            else:
                self.recording = True
                return False
        elif key == 27:
            return True

    def start(self):
        """
        Start the video recorder.
        :return:
        """
        # print("here")
        print("video save file: ", self.save_file)
        if not os.path.exists(self.save_file):
            os.mkdir(self.dataPath)
        self.video = cv2.VideoWriter(self.save_file, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, self.size)

    def record(self, frame):
        """
        Record a frame.
        :param frame: frame to record
        :return:
        """
        print("here")
        vidout = cv2.resize(frame, self.size)
        self.video.write(vidout)
        self.frame_count += 1

    def stop(self):
        """
        Stop the video recorder.
        :return:
        """
        self.video.release()
        self.frame_count = 0

    def save_frame(self, frame, name: str):
        """
        Save a frame to be checked later.
        :param frame: frame to save
        :param name: name of the frame
        :return:
        """
        if not os.path.exists(self.dataPath):
            print("exists")
            os.mkdir(self.dataPath)

        cv2.imwrite(f'{self.save_file}/{name}.jpg', cv2.flip(frame, 1))

    def clear_frame(self):
        """
        Clear the frame.
        :return:
        """
        self.frame_count = 0

    def clear(self):
        """
        Clear the video recorder.
        :return:
        """
        self.video.release()
        self.frame_count = 0
        self.video = None

    def __del__(self):
        """
        Delete the video recorder.
        :return:
        """
        self.clear()

if __name__ == '__main__':
    recorder = VideoRecorder(name='rotate_hand')
    recorder.run()