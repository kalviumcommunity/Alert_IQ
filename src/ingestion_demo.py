"""Run the complete Alert_IQ ingestion validation demonstration."""
from src.ingestion_pipeline import generate_ingestion_report


if __name__ == "__main__":
    print(generate_ingestion_report())
