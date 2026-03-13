import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "未選擇檔案"}), 400

        filename = file.filename
        ext = filename.lower().rsplit(".", 1)[-1]

        mime_map = {
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "tif": "image/tiff",
            "tiff": "image/tiff",
        }
        mime_type = mime_map.get(ext)
        if not mime_type:
            return jsonify({"error": f"不支援的檔案格式: .{ext}"}), 400

        from google.cloud import documentai_v1 as documentai
        from google.cloud import storage
        import vertexai
        from vertexai.generative_models import GenerativeModel

        project_id = os.environ["PROJECT_ID"]
        location = os.environ.get("LOCATION", "us")
        processor_id = os.environ["PROCESSOR_ID"]
        bucket_name = os.environ["GCS_BUCKET"]

        # Read file bytes once for both GCS and Document AI
        file_content = file.read()

        # Verify actual file format via magic bytes
        magic = file_content[:12]
        print(f"DEBUG: filename={filename}, ext={ext}, size={len(file_content)}, magic_hex={magic.hex()}")
        if ext == "png" and magic[:8] != b'\x89PNG\r\n\x1a\n':
            # Check if it's actually HEIC/HEIF
            if b'ftyp' in file_content[:12]:
                return jsonify({"error": "此檔案實際上是 HEIC 格式，非 PNG。請先轉換為 PNG 或 JPEG 後再上傳。"}), 400
            return jsonify({
                "error": "檔案內容與副檔名不符，非有效的 PNG 檔案",
                "debug": {"magic_hex": magic.hex(), "expected": "89504e470d0a1a0a"}
            }), 400

        # 1. Upload to GCS
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_string(file_content, content_type=mime_type)
        gcs_uri = f"gs://{bucket_name}/{filename}"
        print(f"Uploaded to {gcs_uri}")

        # 2. Call Document AI (using raw bytes directly)
        docai_opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
        docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_opts)
        resource_name = docai_client.processor_path(project_id, location, processor_id)

        docai_request = documentai.ProcessRequest(
            name=resource_name,
            raw_document=documentai.RawDocument(content=file_content, mime_type=mime_type),
        )
        result = docai_client.process_document(request=docai_request)

        extracted_data = {}
        for entity in result.document.entities:
            extracted_data[entity.type_] = entity.mention_text

        print(f"Document AI result: {json.dumps(extracted_data, ensure_ascii=False)}")

        # 3. Call Gemini for review (using Vertex AI)
        vertexai.init(project=project_id, location="us-central1")
        model = GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        你現在是一位嚴格但友善的企業財務人員。
        以下是系統剛剛從一張單據中萃取出來的資料（JSON 格式）：
        {json.dumps(extracted_data, ensure_ascii=False)}

        請幫我檢查這份單據，並用繁體中文給出一段「報帳建議評語」。
        請注意以下規則：
        1. 檢查是否有抓到「總金額 (total_amount)」和「日期 (invoice_date)」。
        2. 如果金額超過 3000，請提醒「金額較大，需檢附主管簽核」。
        3. 語氣要專業、簡潔。
        """

        response = model.generate_content(prompt)

        return jsonify({
            "extracted_data": extracted_data,
            "gemini_review": response.text,
        })

    except KeyError as e:
        return jsonify({"error": f"缺少環境變數: {e}"}), 500
    except ImportError as e:
        return jsonify({"error": f"缺少套件，請安裝 requirements-web.txt: {e}"}), 500
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR: {tb}")
        error_detail = {
            "error": f"處理失敗: {str(e)}",
            "debug": {
                "type": type(e).__name__,
                "filename": filename if 'filename' in dir() else None,
                "mime_type": mime_type if 'mime_type' in dir() else None,
                "file_size": len(file_content) if 'file_content' in dir() else None,
                "project_id": project_id if 'project_id' in dir() else None,
                "location": location if 'location' in dir() else None,
                "processor_id": processor_id if 'processor_id' in dir() else None,
                "traceback": tb,
            }
        }
        return jsonify(error_detail), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
