import functions_framework
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json

@functions_framework.cloud_event
def process_invoice(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    gcs_uri = f"gs://{bucket_name}/{file_name}"

    print(f"🚀 偵測到新檔案上傳: {gcs_uri}")

    project_id = os.environ.get("PROJECT_ID")
    location = os.environ.get("LOCATION", "us")
    processor_id = os.environ.get("PROCESSOR_ID")

    # 1. 自動判斷正確的 MIME Type
    mime_type = "application/pdf"
    ext = file_name.lower().split('.')[-1]
    if ext in ['png']:
        mime_type = "image/png"
    elif ext in ['jpg', 'jpeg']:
        mime_type = "image/jpeg"
    elif ext in ['tif', 'tiff']:
        mime_type = "image/tiff"
    else:
        print(f"⚠️ 警告: 未知的副檔名 .{ext}，預設當作 PDF 處理")

    # 2. 從 GCS 下載檔案內容
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    file_content = blob.download_as_bytes()
    print(f"📥 已下載檔案，大小: {len(file_content)} bytes")

    # 3. 呼叫 Document AI (使用 raw_document)
    docai_opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=docai_opts)
    name = client.processor_path(project_id, location, processor_id)

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=file_content, mime_type=mime_type),
    )

    print("🧠 正在呼叫 Document AI 解析發票...")
    result = client.process_document(request=request)
    
    # 整理萃取出來的資料成 JSON 格式
    extracted_data = {}
    for entity in result.document.entities:
        extracted_data[entity.type_] = entity.mention_text
        
    print(f"✅ Document AI 解析完成：{json.dumps(extracted_data, ensure_ascii=False)}")

    # 3. 呼叫 Vertex AI (Gemini) 進行智能審核
    print("✨ 正在交給 Gemini 撰寫報帳評語...")
    vertexai.init(project=project_id, location="us-central1")
    # 使用最新的 Gemini 1.5 Flash 模型 (快速且便宜)
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
    print("=========================================")
    print(f"🤖 Gemini 智能評語：\n{response.text}")
    print("=========================================")
    
    print("🎉 流程結束！")