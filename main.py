import os
import functions_framework
from google.cloud import documentai_v1 as documentai


@functions_framework.cloud_event
def process_invoice(cloud_event):
    """Cloud Function triggered by Cloud Storage finalize event.

    Processes an uploaded invoice using Document AI and logs extracted entities.
    """
    data = cloud_event.data
    bucket = data["bucket"]
    name = data["name"]
    gcs_uri = f"gs://{bucket}/{name}"

    print(f"Processing file: {gcs_uri}")

    project_id = os.environ["PROJECT_ID"]
    location = os.environ["LOCATION"]
    processor_id = os.environ["PROCESSOR_ID"]

    client = documentai.DocumentProcessorServiceClient()
    resource_name = client.processor_path(project_id, location, processor_id)

    gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type="application/pdf")
    request = documentai.ProcessRequest(
        name=resource_name,
        gcs_document=gcs_document,
    )

    result = client.process_document(request=request)
    document = result.document

    print(f"Document text (first 200 chars): {document.text[:200]}")

    for entity in document.entities:
        print(
            f"Entity: type={entity.type_}, "
            f"mention_text={entity.mention_text}, "
            f"confidence={entity.confidence:.2%}"
        )

    print(f"Total entities extracted: {len(document.entities)}")
