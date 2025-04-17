import subprocess
from typing import Dict, List, Tuple, Union

from loguru import logger


def scan_model_with_modelscan(model_path: str) -> int:
    """
    Scan the model using ModelScan from ProtectAI.

    Args:
        model_path (str): Path to the model file

    Returns:
        int: Exit code from the scan
    """
    logger.info(f"Scanning model at {model_path} with ModelScan...")

    # Run modelscan command
    try:
        result = subprocess.run(
            ["modelscan", "-p", model_path],
            capture_output=True,
            text=True,
            check=False
        )

        # Print the output
        logger.info("\nScan Results:")
        logger.info(result.stdout)

        if result.stderr:
            logger.info("Errors:")
            logger.info(result.stderr)

        logger.info(f"Scan exit code: {result.returncode}")

        # Interpret exit code
        if result.returncode == 0:
            logger.info("✓ Scan completed successfully, no vulnerabilities found")
        elif result.returncode == 1:
            logger.info("⚠ Scan completed successfully, but vulnerabilities were found")
        elif result.returncode == 2:
            logger.error("✗ Scan failed, ModelScan threw an error while scanning")
        elif result.returncode == 3:
            logger.error("✗ No supported files were passed to the tool")
        elif result.returncode == 4:
            logger.error("✗ Usage error, CLI was passed invalid or incomplete options")

        return result.returncode

    except FileNotFoundError:
        logger.error("ModelScan is not installed or not in PATH")
        logger.info("Please install it using: pip install modelscan")
        return -1


def scan_multiple_files(file_paths: List[str]) -> Dict[str, int]:
    """
    Scan multiple model-related files with ModelScan.

    Args:
        file_paths (List[str]): List of file paths to scan

    Returns:
        Dict[str, int]: Dictionary mapping file paths to their scan exit codes
    """
    results = {}
    overall_status = True

    logger.info(f"Starting security scan of {len(file_paths)} files...")

    for file_path in file_paths:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Scanning: {file_path}")
        logger.info(f"{'=' * 50}")

        result_code = scan_model_with_modelscan(file_path)
        results[file_path] = result_code

        # Consider any non-zero exit code (except for not installed) as a potential issue
        if result_code > 0:
            overall_status = False

    # Print summary
    logger.info("\n\n" + "=" * 60)
    logger.info("SECURITY SCAN SUMMARY")
    logger.info("=" * 60)

    for file_path, code in results.items():
        status = "✓ PASS" if code == 0 else "⚠ FAIL" if code > 0 else "❌ ERROR"
        logger.info(f"{status}: {file_path}")

    logger.info(f"\nOverall status: {'PASS' if overall_status else 'ISSUES DETECTED'}")
    logger.info("=" * 60)

    return results


def scan_model_and_processor(model_path: str, processor_path: str) -> Tuple[bool, Dict[str, int]]:
    """
    Scan both model and processor files for security issues.

    Args:
        model_path (str): Path to the model file
        processor_path (str): Path to the processor/feature extractor file

    Returns:
        Tuple[bool, Dict[str, int]]: (overall_pass_status, results_by_file)
    """
    files_to_scan = [model_path, processor_path]
    results = scan_multiple_files(files_to_scan)

    # Check if all scans passed (code 0) or not installed (code -1)
    overall_pass = all(code <= 0 for code in results.values())

    return overall_pass, results


def verify_model_security(model_path, processor_path):
    """
    Verify the security of the model and processor files.

    Args:
        model_path (str): Path to the model file
        processor_path (str): Path to the processor file

    Returns:
        bool: True if security verification passed, False otherwise
    """
    logger.info("Scanning model and processor for security vulnerabilities...")
    overall_safe, scan_results = scan_model_and_processor(model_path, processor_path)

    if not overall_safe:
        logger.warning("Security issues detected in model files!")
        user_input = input("Security vulnerabilities were found. Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Exiting due to security concerns.")
            return False
        logger.warning("Continuing despite security concerns...")
    else:
        logger.success("Security scan completed successfully. No vulnerabilities found.")

    return True
