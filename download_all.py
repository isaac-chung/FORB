import glob
import os
from subprocess import Popen

for fname in glob.glob("metadata/*.ndjson"):
    sub = Popen(
      ["python", "downloader/download_images.py", "--metadata_file_path", fname, "--output_path", "saved_images"],
      cwd=os.getcwd())
    (output, _) = sub.communicate()
