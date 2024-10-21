import csv


class SummaryLogger:
    def __init__(self, summary_csv_path):
        self.summary_csv_path = summary_csv_path
        self.summary_header = ['filename', 'total_frames', 'percentage_0_faces', 'percentage_1_face', 'percentage_multiple_faces']
        self._initialize_file()

    def _initialize_file(self):
        with open(self.summary_csv_path, mode='w', newline='') as summary_file:
            summary_writer = csv.writer(summary_file)
            summary_writer.writerow(self.summary_header)

    def log_summary(self, face_counts, filename):
        total_frames = len(face_counts)
        if total_frames == 0:
            print("SUMMARY: No frames to process.")
            return

        zero_faces = sum(1 for count in face_counts.values() if count == 0)
        one_face = sum(1 for count in face_counts.values() if count == 1)
        multiple_faces = total_frames - zero_faces - one_face

        zero_faces_pct = (zero_faces / total_frames) * 100
        one_face_pct = (one_face / total_frames) * 100
        multiple_faces_pct = (multiple_faces / total_frames) * 100

        print(f"SUMMARY: Total frames: {total_frames}")
        print(f"SUMMARY: Frames with 0 faces: {zero_faces} ({zero_faces_pct:.2f}%)")
        print(f"SUMMARY: Frames with 1 face: {one_face} ({one_face_pct:.2f}%)")
        print(f"SUMMARY: Frames with multiple faces: {multiple_faces} ({multiple_faces_pct:.2f}%)")

        with open(self.summary_csv_path, mode='a', newline='') as summary_file:
            summary_writer = csv.writer(summary_file)
            summary_writer.writerow([filename, total_frames, zero_faces_pct, one_face_pct, multiple_faces_pct])