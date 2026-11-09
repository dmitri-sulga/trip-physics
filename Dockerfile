# trip-physics demo container.
#
# The harness makes NO network calls at runtime. Every dependency is baked in
# at build time; the container can (and should) be run with all egress blocked:
#
#   docker build -t trip-physics .
#   mkdir -p out
#   docker run --rm --network=none -v "$PWD/out:/app/out" trip-physics
#
# If it works with --network=none, the no-egress claim is enforced, not promised.

# Pinned to a patch version rather than the floating `python:3.11-slim`: a
# harness whose selling point is reproducibility should not silently rebuild
# against a different base image next month. For a fully immutable build, pin
# the digest instead: FROM python:3.11.14-slim@sha256:<digest>
FROM python:3.11.14-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY schema ./schema
COPY tests ./tests

RUN pip install --no-cache-dir .

# Writes ./out/report.json and ./out/report.md inside the container;
# mount a volume on /app/out to keep them.
CMD ["python", "examples/run_demo.py"]
