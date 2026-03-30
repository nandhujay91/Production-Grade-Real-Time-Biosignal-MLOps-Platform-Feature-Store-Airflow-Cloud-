import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 🔥 Base processed directory
BASE_PROCESSED_DIR = "data/processed"

# 🔥 Signal configuration
signal_info = {
    'Aux': {'dtype': np.int16, 'channels': 3},
    'IMU': {'dtype': np.int16, 'channels': 9},
    'Ephy': {'dtype': np.int16, 'channels': 8},
    'Oxym': {'dtype': np.int32, 'channels': 2},
}


# 🔍 Detect signal type from filename
def detect_signal_type(filename):
    for sig_type in signal_info:
        if sig_type.lower() in filename.lower():
            return sig_type
    raise ValueError(f"❌ Unknown signal type: {filename}")


# 📥 Load binary file
def load_bin(filepath, dtype, channels):
    logger.info(f"📥 Loading file: {filepath}")

    try:
        data = np.fromfile(filepath, dtype=dtype)
    except Exception as e:
        logger.error(f"❌ Failed to read file: {e}")
        raise

    if len(data) == 0:
        raise ValueError(f"❌ Empty file: {filepath}")

    if len(data) % channels != 0:
        logger.warning(f"⚠️ Truncating extra data for {filepath}")
        data = data[:len(data) - (len(data) % channels)]

    data = data.reshape(-1, channels)

    if np.isnan(data).any():
        raise ValueError(f"❌ NaN detected in {filepath}")

    logger.info(f"✅ Loaded shape: {data.shape}")
    return data


# 🔄 Process single file
def process_single_file(session_id, filename, path):
    logger.info(f"🚀 Processing file: {filename}")

    try:
        sig_type = detect_signal_type(filename)
        info = signal_info[sig_type]

        data = load_bin(path, info["dtype"], info["channels"])

        # 🔥 Channel selection (UNCHANGED)
        if sig_type == "Ephy":
            data = data[:, [0]]
            columns = ["ecg_ch0"]

        elif sig_type == "IMU":
            data = data[:, :3]
            data = data[np.any(data != 0, axis=1)]
            columns = ["acc_x", "acc_y", "acc_z"]

        elif sig_type == "Aux":
            columns = [f"aux_ch{i}" for i in range(data.shape[1])]

        elif sig_type == "Oxym":
            columns = [f"oxym_ch{i}" for i in range(data.shape[1])]

        else:
            columns = None

        # 🔥 Create session folder
        session_dir = os.path.join(BASE_PROCESSED_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 🔥 Save CSV
        csv_name = filename.replace(".bin", ".csv")
        output_path = os.path.join(session_dir, csv_name)

        df = pd.DataFrame(data, columns=columns)

        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            logger.error(f"❌ Failed to save CSV: {e}")
            raise

        output_path = output_path.replace("\\", "/")

        logger.info(f"📤 Saved: {output_path}")
        logger.info(f"📊 Final shape: {data.shape}")

        return {
            "type": sig_type,
            "output": output_path,
            "shape": data.shape
        }

    except Exception as e:
        logger.error(f"❌ Failed processing file {filename}: {e}", exc_info=True)
        raise


# 🔥 Main processing function
def process_files(session_id, files):
    logger.info("===================================")
    logger.info(f"🧠 Starting processing for session: {session_id}")
    logger.info(f"⏱ Time: {datetime.now()}")
    logger.info("===================================")

    results = []

    for f in files:
        try:
            result = process_single_file(
                session_id,
                f["filename"],
                f["path"]
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"⚠️ Skipping file due to error: {e}")

    logger.info(f"✅ Processing completed for session: {session_id}")
    logger.info("===================================")

    return results