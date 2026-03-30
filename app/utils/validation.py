ALLOWED_EXTENSIONS = [".bin"]
MAX_FILE_SIZE_MB = 50


def validate_file(file):
    filename = file.filename

    # ✅ Extension check
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, "Only .bin files are allowed."

    # ✅ File size check
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    size_mb = size / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.2f} MB)"

    if size == 0:
        return False, "Empty file"

    return True, "Valid file"