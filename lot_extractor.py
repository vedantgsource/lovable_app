def run_lot_extraction(order_pdf_path, platmap_pdf_path):
    import os
    import json
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from PyPDF2 import PdfReader
    from openai import OpenAI
    from rich.console import Console
    from pdf2image import convert_from_path
    from ultralytics import YOLO
    import easyocr

    # === CONFIGURATION ===
    MODEL_PATH = "Well_best.pt"
    POPPLER_PATH = r"C:\Users\30004\poppler\poppler-24.08.0\Library\bin"  # Change for local, use /usr/bin/poppler or auto for Docker
    OPENROUTER_API_KEY = "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"  # Insert key or load from env
    MODEL = "mistralai/mistral-7b-instruct:free"

    TILE_SIZE = 640
    OVERLAP = 0
    ZOOM_OUT_PADDING = 50
    OUTPUT_DIR = "output_predictions"
    SNAPSHOT_DIR = "snapshots"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    console = Console()

    # === STEP 1: Extract lot_number & elevation from OrderForm ===
    def extract_text_from_pdf(pdf_path):
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    console.print(f"[bold yellow]📄 Reading Order Form:[/bold yellow] {order_pdf_path}")
    pdf_text = extract_text_from_pdf(order_pdf_path)

    prompt = f"""
    You are an intelligent parser. From the following text extracted from a sales order form PDF, extract:

    - "lot_number": the Lot Number (e.g. 44, 99, etc.), take number only
    - "elevation": the Elevation or Floorplan ID (e.g. 1827/H, 2605/B, etc.), take alphabet only

    Return only this JSON format:
    {{
       "lot_number": "<value>",
       "elevation": "<value>"
    }}

    If not found, use null for the value.

    Text:
    --------------------
    {pdf_text}
    --------------------
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    try:
        extracted_json = json.loads(response.choices[0].message.content)
    except Exception:
        extracted_json = {"lot_number": None, "elevation": None}

    target_lot_number = extracted_json.get("lot_number")
    if not target_lot_number:
        return None

    # === STEP 2: Detect lot in PlatMap ===
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False)

    lot_number_class_id = next((cls for cls, name in model.names.items() if "lot_number" in name.lower()), None)
    lot_form_class_id = next((cls for cls, name in model.names.items() if "lot_form" in name.lower()), None)

    pages = convert_from_path(platmap_pdf_path, dpi=300, poppler_path=POPPLER_PATH)

    def tile_image(image, tile_size, overlap=0):
        h, w = image.shape[:2]
        tiles, positions = [], []
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                tile = image[y:min(y+tile_size, h), x:min(x+tile_size, w)]
                tiles.append(tile)
                positions.append((x, y))
        return tiles, positions

    snapshot_path = None

    for i, page in enumerate(pages):
        page_np = np.array(page.convert("RGB"))
        image = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
        tiles, positions = tile_image(image, TILE_SIZE, OVERLAP)
        results = [model(tile)[0] for tile in tiles]

        for tile, (x, y), result in zip(tiles, positions, results):
            for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(cls_id) != lot_number_class_id:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                gx1, gy1, gx2, gy2 = x + x1, y + y1, x + x2, y + y2
                roi = image[gy1:gy2, gx1:gx2]
                ocr_result = reader.readtext(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

                for (_, text, conf) in ocr_result:
                    if conf > 0.4 and text.strip() == target_lot_number:
                        pad = 600
                        crop_x1, crop_y1 = max(gx1 - pad, 0), max(gy1 - pad, 0)
                        crop_x2, crop_y2 = min(gx2 + pad, image.shape[1]), min(gy2 + pad, image.shape[0])
                        cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                        snapshot_path = os.path.join(SNAPSHOT_DIR, f"lot_{text.strip()}.png")
                        cv2.imwrite(snapshot_path, cropped_img)
                        break
            if snapshot_path:
                break
        if snapshot_path:
            break

    if not snapshot_path:
        return None

    # === STEP 3: Refine to exact boundary ===
    img = cv2.imread(snapshot_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)[0]

    lot_number_boxes = []
    lot_form_boxes = []
    target_lot_center = None
    target_lot_bbox = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        label = model.names[cls_id]
        if label == "lot_number":
            lot_number_boxes.append((x1, y1, x2, y2))
        elif label == "lot_form":
            lot_form_boxes.append((x1, y1, x2, y2))

    for (x1, y1, x2, y2) in lot_number_boxes:
        cropped = img_rgb[y1:y2, x1:x2]
        result = reader.readtext(cropped)
        for (_, text, conf) in result:
            if text.strip() == target_lot_number:
                target_lot_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    if target_lot_center:
        min_dist = float("inf")
        for (fx1, fy1, fx2, fy2) in lot_form_boxes:
            f_center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)
            dist = np.linalg.norm(np.array(target_lot_center) - np.array(f_center))
            if dist < min_dist:
                min_dist = dist
                target_lot_bbox = (fx1, fy1, fx2, fy2)

    if target_lot_bbox:
        tx1, ty1, tx2, ty2 = target_lot_bbox
        tx1, ty1 = max(0, tx1 - ZOOM_OUT_PADDING), max(0, ty1 - ZOOM_OUT_PADDING)
        tx2, ty2 = min(img.shape[1], tx2 + ZOOM_OUT_PADDING), min(img.shape[0], ty2 + ZOOM_OUT_PADDING)
        final_crop = img[ty1:ty2, tx1:tx2]
        output_path = os.path.join("output_predictions", f"final_cropped.png")
        cv2.imwrite(output_path, final_crop)
        return output_path
    else:
        return snapshot_path
