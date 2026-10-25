import asyncio
from contextlib import suppress


async def cancel_worker_task(task):
    """Cancel and await an optional deployment worker task."""
    if task is None:
        return None

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    return None
