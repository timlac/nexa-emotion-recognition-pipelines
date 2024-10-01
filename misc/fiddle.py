import os
sensitive_data_path = os.getenv("SENSITIVE_DATA_DIR")

print(sensitive_data_path)

# with open(os.path.join(sensitive_data_path, '/kosmos/split/KOSMOS077_IS_LSI_LEFT.mp4'), 'r') as f:
#     data = f.read()