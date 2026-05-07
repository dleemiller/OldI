# Audiveris 5.x OMR in a Podman image. Runs as a fallback to Clarity-OMR.
#
# Build:   podman build -t oldi-audiveris -f src/oldi/containers/audiveris.Containerfile .
# Run:     podman run --rm -v $PWD/data:/data:Z oldi-audiveris \
#              -batch -export -output /data/05_musicxml/<book>/... \
#              /data/03_tunes/<book>/page_XXXX/tune_NN/crop.pdf

FROM docker.io/eclipse-temurin:25-jdk AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ca-certificates tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/Audiveris/audiveris.git . \
 && ./gradlew --no-daemon -x test build

FROM docker.io/eclipse-temurin:25-jdk

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng \
      libfreetype6 fontconfig \
      libgtk-3-0 libglib2.0-0 libxtst6 \
 && rm -rf /var/lib/apt/lists/*

# Audiveris >=5.6 uses a multi-project Gradle layout; the distributable
# tarball lands under app/build/distributions/. Expanded dir is `app-<ver>/`.
COPY --from=build /src/app/build/distributions/ /opt/audiveris/
RUN cd /opt/audiveris \
 && tar xf app-*.tar \
 && rm -f app-*.tar app-*.zip app-*.tar.gz \
 && mv app-* audiveris

ENV PATH="/opt/audiveris/audiveris/bin:${PATH}"
# Audiveris 5.10+ renamed the launcher from `Audiveris` to `app`; fall back to
# whichever exists.
RUN ls /opt/audiveris/audiveris/bin/
ENTRYPOINT ["/bin/bash", "-c", "exec $(ls /opt/audiveris/audiveris/bin/Audiveris /opt/audiveris/audiveris/bin/app 2>/dev/null | head -1) \"$@\"", "--"]
