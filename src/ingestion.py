import os
import io
import base64
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.docstore.document import Document
import camelot
from openai import OpenAI


def _resize_for_vision(image_bytes: bytes, max_side: int = 1400, jpeg_quality: int = 85) -> bytes:
    """Downscale & convert to JPEG to keep payload small and reliable."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out.getvalue()


class DocumentLoader:
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment.")
        self.client = OpenAI(api_key=key)

    def load_file(self, file, filename):
        ext = filename.lower().split(".")[-1]
        file.seek(0)  # Ensure file pointer is at the beginning
        file_bytes = file.read()

        text_docs, tables, vision_items = [], [], []

        if ext == "pdf":
            text_docs, tables, vision_items = self._process_pdf(file_bytes, filename)
        elif ext in ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]:
            tdocs, vitems = self._process_image(file_bytes, filename)
            text_docs.extend(tdocs)
            vision_items.extend(vitems)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return {"text_docs": text_docs, "tables": tables, "vision": vision_items}

    def _process_pdf(self, file_bytes, filename):
        text_docs, tables, vision_items = [], [], []

        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            for p_num, page in enumerate(pdf):
                p = p_num + 1
                combined_content = []

                text = page.get_text()
                if text and text.strip():
                    combined_content.append(f"[TEXT OCR]\n{text.strip()}")

                for img_idx, info in enumerate(page.get_images(full=True), start=1):
                    xref = info[0]
                    base_image = pdf.extract_image(xref)
                    img_bytes = base_image.get("image", b"")
                    if not img_bytes:
                        continue

                    desc, struct, err = self._analyze_image(img_bytes)
                    if err:
                        combined_content.append(f"[VISION ERROR] {err}")
                    if desc:
                        combined_content.append(f"[VISION ANALYSIS]\n{desc}")
                    if struct:
                        vision_items.append({
                            "page": p, "figure": img_idx, "source": filename, "data": struct
                        })

                if combined_content:
                    text_docs.append(Document(
                        page_content="\n\n".join(combined_content).strip(),
                        metadata={"page": p, "source": filename}
                    ))

            pdf.close()

            try:
                # Camelot needs a file path, so we write to a temp file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tbls = camelot.read_pdf(tmp.name, pages="all", flavor='lattice')
                    for t in tbls:
                        tables.append(t.df)
            except Exception:
                pass  # Table extraction is best-effort

        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")

        return text_docs, tables, vision_items

    def _process_image(self, img_bytes, filename):
        text_docs, vision_items = [], []

        try:
            raw_img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(raw_img) or ""
        except Exception:
            ocr_text = ""

        desc, struct, err = self._analyze_image(img_bytes)

        combined_content = []
        if ocr_text.strip():
            combined_content.append(f"[TEXT OCR]\n{ocr_text.strip()}")
        if err:
            combined_content.append(f"[VISION ERROR] {err}")
        if desc:
            combined_content.append(f"[VISION ANALYSIS]\n{desc}")

        if not combined_content:
            combined_content.append("[VISION ANALYSIS] (No visible text and vision analysis returned no description.)")

        text_docs.append(Document(
            page_content="\n\n".join(combined_content).strip(),
            metadata={"page": 1, "source": filename}
        ))

        if struct:
            vision_items.append({
                "page": 1, "figure": 1, "source": filename, "data": struct
            })

        return text_docs, vision_items

    def _analyze_image(self, image_bytes: bytes):
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                print(f"DEBUG: Analyzing image with original dimensions: {w}x{h}")

            compact_bytes = _resize_for_vision(image_bytes)
            b64_image = base64.b64encode(compact_bytes).decode("utf-8")

            image_payload = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            }

            desc, struct = None, {}
            try:
                r1 = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                         "content": "Describe images precisely. Include objects, text, numbers, colors, and relationships."},
                        {"role": "user", "content": [
                            {"type": "text",
                             "text": "Describe this image in 7-10 bullet points. If it's a chart/graph, include its type, labels, values/percentages, and key takeaways like the largest/smallest segment. If it's a photo, mention key objects, actions, scene, and notable text."},
                            image_payload
                        ]}
                    ],
                    max_tokens=400
                )
                desc = (r1.choices[0].message.content or "").strip()
            except Exception as e:
                return None, None, f"Vision summary failed: {e}"

            try:
                r2 = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": "If the image is a chart (pie, bar, line, etc.), extract its data into a structured JSON object. Use this schema: {chart_type: string|null, title: string|null, slices: [{label: string|null, percent: number|null, value: number|null, color: {name: string|null, hex: string|null}}]}. If it is NOT a chart, you must return an empty JSON object {}."},
                        {"role": "user", "content": [
                            {"type": "text",
                             "text": "Extract structured data from this chart. If it is not a chart, return {}."},
                            image_payload
                        ]}
                    ],
                    max_tokens=800
                )
                raw_json = r2.choices[0].message.content
                if raw_json:
                    struct = json.loads(raw_json)
            except Exception:
                pass  # Structured extraction is best-effort

            return desc, struct, None

        except Exception as outer_e:
            return None, None, f"Vision pipeline failed: {outer_e}"