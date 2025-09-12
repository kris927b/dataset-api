import aio_pika
from pathlib import Path

from ..core.config import settings


def check_root_dir_exists() -> tuple[bool, str]:
    """Checks if the configured ROOT_DIR exists and is a directory."""
    root_path = Path(settings.ROOT_DIR)
    if root_path.exists() and root_path.is_dir():
        return True, f"Root directory '{root_path}' is accessible."
    else:
        return False, f"Root directory '{root_path}' not found or is not a directory."


async def check_rabbitmq_connection() -> tuple[bool, str]:
    """Checks if a connection to RabbitMQ can be established."""
    try:
        connection = await aio_pika.connect_robust(settings.RABBITMQ_URL, timeout=5)
        await connection.close()
        return True, "RabbitMQ connection successful."
    except Exception as e:
        return False, f"Failed to connect to RabbitMQ: {e}"
