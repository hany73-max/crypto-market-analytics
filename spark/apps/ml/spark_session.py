"""
==============================================================================
spark/apps/ml/spark_session.py
Unified SparkSession factory with cross-platform (Windows & Linux/Docker) support.

Handles Windows Hadoop home, winutils, PySpark worker Python executable paths,
and standard performance/deterministic configurations.
==============================================================================
"""

import logging
import os
import platform
import sys
from typing import Optional

from pyspark.sql import SparkSession

logger = logging.getLogger("crypto_ml_spark")


def configure_windows_environment(project_root: Optional[str] = None) -> None:
    """Configure environment variables required for running PySpark safely on Windows."""
    if platform.system().lower() != "windows":
        return

    # Ensure worker processes invoke the current virtualenv Python
    python_exe = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

    # Resolve HADOOP_HOME if unset
    if not os.environ.get("HADOOP_HOME"):
        if project_root is None:
            # Walk up to repo root
            current_dir = os.path.abspath(os.path.dirname(__file__))
            # from spark/apps/ml -> root is 3 levels up
            project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

        candidate_hadoop = os.path.join(project_root, "data", "hadoop")
        if os.path.exists(candidate_hadoop):
            os.environ["HADOOP_HOME"] = candidate_hadoop
            bin_path = os.path.join(candidate_hadoop, "bin")
            if bin_path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_path + ";" + os.environ.get("PATH", "")
            logger.debug("Configured Windows HADOOP_HOME: %s", candidate_hadoop)


def get_spark_session(
    app_name: str = "CryptoMLPipeline",
    master: Optional[str] = None,
    log_level: str = "WARN",
) -> SparkSession:
    """
    Build or retrieve a configured SparkSession.

    Args:
        app_name: Name of the Spark application.
        master: Spark master URL (e.g., 'local[*]', 'spark://...'). Defaults to local[*] if unset.
        log_level: Spark log level ('ALL', 'DEBUG', 'ERROR', 'FATAL', 'INFO', 'OFF', 'WARN').

    Returns:
        Configured SparkSession instance.
    """
    configure_windows_environment()

    resolved_master = master or os.environ.get("SPARK_MASTER", "local[*]")

    builder = (
        SparkSession.builder.appName(app_name)
        .master(resolved_master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
    )

    if "local" in resolved_master:
        builder = (
            builder.config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark
