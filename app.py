import os
import json
from flask import Flask, render_template, request, jsonify
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
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

    project_id = os.environ["PROJECT_ID"]
    location = os.environ.get("LOCATION", "us")
    processor_id = os.environ["PROCESSOR_ID"]
    bucket_name = os.environ["GCS_BUCKET"]

    # 1. Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_file(file, content_type=mime_type)
    gcs_uri = f"gs://{bucket_name}/{filename}"
    print(f"Uploaded to {gcs_uri}")

    # 2. Call Document AI
    docai_client = documentai.DocumentProcessorServiceClient()
    resource_name = docai_client.processor_path(project_id, location, processor_id)

    docai_request = documentai.ProcessRequest(
        name=resource_name,
        gcs_document=documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type),
    )
    result = docai_client.process_document(request=docai_request)

    extracted_data = {}
    for entity in result.document.entities:
        extracted_data[entity.type_] = entity.mention_text

    print(f"Document AI result: {json.dumps(extracted_data, ensure_ascii=False)}")

    # 3. Call Gemini for review
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001")

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
