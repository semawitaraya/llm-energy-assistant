import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting ingestion pipeline...")
    t0 = time.perf_counter()

    from src.rag_pipeline import get_or_create_vectorstore
    vectorstore, n = get_or_create_vectorstore()

    elapsed = time.perf_counter() - t0
    logger.info("Done! %d chunks indexed in %.1fs", n, elapsed)
    logger.info("Index saved to data/vectorstore/")


if __name__ == "__main__":
    main()