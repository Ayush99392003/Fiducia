import os
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHttp

def init_tracer(session_id: str):
    """
    Initializes the OpenTelemetry TracerProvider to send traces to Arize Phoenix.
    Includes the session identifier and project metadata.
    """
    # Read the Phoenix collector endpoint from environment or default to local Phoenix
    # Using HTTP exporter as Phoenix typically listens on 6006 for HTTP OTLP
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    phoenix_api_key = os.getenv("PHOENIX_API_KEY")

    # If the user mistakenly passed the Phoenix Cloud UI URL instead of the collector URL, fix it
    if "app.phoenix.arize.com/s/" in phoenix_endpoint:
        phoenix_endpoint = "https://app.phoenix.arize.com/v1/traces"

    headers = {}
    if phoenix_api_key:
        headers["Authorization"] = f"Bearer {phoenix_api_key}"

    # Quick test to see if endpoint is reachable and authorized
    import urllib.request
    import urllib.error
    try:
        # We send an empty POST. If unauthorized, it returns 401.
        # If authorized, it might return 415 or 400 (since body is empty), which means we are connected.
        req = urllib.request.Request(phoenix_endpoint, method="POST", headers=headers)
        urllib.request.urlopen(req, timeout=3)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print(f"Warning: Phoenix tracing unauthorized ({e.code}). Tracing disabled. Continuing without tracing...")
            return trace_api.get_tracer(__name__)
    except Exception as e:
        print(f"Warning: Phoenix tracing endpoint unreachable ({e}). Tracing disabled. Continuing without tracing...")
        return trace_api.get_tracer(__name__)

    provider = trace_sdk.TracerProvider(
        resource=Resource(
            attributes={
                "model_id": "hackerrank-orchestrate-agent",
                "model_version": "v1.0",
                "session.id": session_id,
            }
        )
    )

    # Configure the OTLP HTTP exporter
    exporter = OTLPSpanExporterHttp(endpoint=phoenix_endpoint, headers=headers)
    
    # Add the span processor to the provider
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    
    # Register the provider globally
    trace_api.set_tracer_provider(provider)
    
    return trace_api.get_tracer(__name__)
