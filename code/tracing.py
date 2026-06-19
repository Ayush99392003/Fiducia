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
    exporter = OTLPSpanExporterHttp(endpoint=phoenix_endpoint)
    
    # Add the span processor to the provider
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    
    # Register the provider globally
    trace_api.set_tracer_provider(provider)
    
    return trace_api.get_tracer(__name__)
