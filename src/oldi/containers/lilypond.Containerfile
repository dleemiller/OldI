FROM docker.io/library/debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      lilypond \
      python3 \
      python3-pip \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Convention: caller mounts input and output dirs; we invoke musicxml2ly + lilypond.
# Usage:
#   podman run --rm \
#       -v $PWD/data:/data:Z \
#       oldi-lilypond \
#       /data/06_validated/book/page_0010/tune_01.musicxml /data/08_pdf/book/page_0010/tune_01.pdf

ENTRYPOINT ["/bin/bash", "-eu", "-c", "\
  SRC=\"$0\"; DST=\"$1\"; \
  WORK=$(mktemp -d); \
  cp \"$SRC\" \"$WORK/in.musicxml\"; \
  cd \"$WORK\"; \
  musicxml2ly -o out.ly in.musicxml; \
  lilypond --pdf -dno-point-and-click -o out out.ly; \
  mkdir -p \"$(dirname \"$DST\")\"; \
  mv out.pdf \"$DST\"; \
"]
