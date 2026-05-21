"""
Production entry-point for Render (and any environment that injects PORT).

Problem: uvicorn's startup sequence runs the ASGI lifespan *before* binding
the socket.  On Render the ML warm-up inside lifespan takes several minutes,
so the port is never open when Render's TCP probe fires → "No open ports
detected".

Fix: bind the socket at the OS level here, before handing it to uvicorn.
The kernel's TCP stack accepts SYN packets (Render's port probe) immediately;
uvicorn picks up queued connections after lifespan completes.

Local behaviour is unchanged — PORT defaults to 8000 when unset.
"""
import os
import socket

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    # Bind and listen at OS level so Render's TCP probe succeeds during
    # the ML model warm-up that runs inside uvicorn's lifespan.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(128)

    config = uvicorn.Config("app.main:app", host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
