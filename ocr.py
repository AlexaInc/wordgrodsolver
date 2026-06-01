import easyocr
import easyocr.easyocr
import sys
import json
import os
from PIL import Image
import io

# Fix EasyOCR bug
easyocr.easyocr.corrupt_msg = "Model error"

def process_image(reader, image_path):
    try:
        if not os.path.exists(image_path):
            return {"error": f"Path not found: {image_path}"}

        # Single-pass OCR is much faster and usually accurate enough for Deep Learning models
        results = reader.readtext(image_path, detail=1)
        if not results:
             return {"error": "No text detected"}

        symbols = []
        for (bbox, text, prob) in results:
            text = text.strip().upper()
            clean_text = "".join([c for c in text if c.isalpha()])
            if not clean_text: continue
            
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            
            if len(clean_text) > 1:
                # Approximate positions for concatenated chars
                w = bbox[2][0] - bbox[0][0]
                char_w = w / len(clean_text)
                for i, char in enumerate(clean_text):
                    symbols.append({
                        "text": char,
                        "x": bbox[0][0] + (i + 0.5) * char_w,
                        "y": y_center
                    })
            else:
                symbols.append({"text": clean_text, "x": x_center, "y": y_center})

        if not symbols: return []

        # Find 8 distinct lanes for X and Y
        def get_lanes(coords, num_lanes=8):
            coords.sort()
            if not coords: return []
            lanes = []
            # Simple clustering: divide range into 8 buckets
            mi, ma = min(coords), max(coords)
            if ma == mi: return [mi]
            
            bucket_size = (ma - mi) / (num_lanes - 1) if num_lanes > 1 else 1
            for i in range(num_lanes):
                center = mi + i * bucket_size
                lanes.append(center)
            return lanes

        xs = [s["x"] for s in symbols]
        ys = [s["y"] for s in symbols]
        
        x_lanes = get_lanes(xs, 8)
        y_lanes = get_lanes(ys, 8)
        
        grid = [[" " for _ in range(8)] for _ in range(8)]
        for s in symbols:
            # Map to nearest lane
            r = min(range(8), key=lambda i: abs(s["y"] - y_lanes[i]))
            c = min(range(8), key=lambda i: abs(s["x"] - x_lanes[i]))
            grid[r][c] = s["text"]

        final_symbols = []
        for r in range(8):
            for c in range(8):
                if grid[r][c] != " ":
                    final_symbols.append({"text": grid[r][c], "r": r, "c": c})
        return final_symbols
                    
    except Exception as e:
        return {"error": str(e)}

def main():
    # Initialize reader ONCE
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        # Signal ready
        print("READY", flush=True)
    except Exception as e:
        print(json.dumps({"error": f"Init failed: {str(e)}"}), flush=True)
        return

    # Listen for image paths on stdin
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        image_path = line.strip()
        if not image_path:
            continue
            
        result = process_image(reader, image_path)
        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
