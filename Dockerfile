# ============================================================
# Mnemo — Multi-stage Docker build
# Target: single static binary, ~100 MB final image
# ============================================================

# --------------- stage 1: build ---------------
FROM rust:1.83-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        cmake \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY crates/mnemo-proxy/Cargo.toml   crates/mnemo-proxy/Cargo.toml
COPY crates/mnemo-cache/Cargo.toml   crates/mnemo-cache/Cargo.toml
COPY crates/mnemo-intelligence/Cargo.toml crates/mnemo-intelligence/Cargo.toml
COPY crates/mnemo-mcp/Cargo.toml     crates/mnemo-mcp/Cargo.toml
COPY crates/mnemo-acp/Cargo.toml     crates/mnemo-acp/Cargo.toml

# Create dummy source files so cargo can resolve the workspace and
# download + compile dependencies (cache-friendly).
RUN mkdir -p crates/mnemo-proxy/src && echo 'fn main() {}' > crates/mnemo-proxy/src/main.rs \
    && mkdir -p crates/mnemo-cache/src && echo '' > crates/mnemo-cache/src/lib.rs \
    && mkdir -p crates/mnemo-intelligence/src && echo '' > crates/mnemo-intelligence/src/lib.rs \
    && mkdir -p crates/mnemo-mcp/src && echo '' > crates/mnemo-mcp/src/lib.rs \
    && mkdir -p crates/mnemo-acp/src && echo '' > crates/mnemo-acp/src/lib.rs

RUN cargo build --release --bin mnemo 2>/dev/null || true

# Now copy the real source and rebuild (only changed crates recompile).
COPY crates/ crates/
# Touch all source files so cargo sees them as newer than the dummies.
RUN find crates -name '*.rs' -exec touch {} +

RUN cargo build --release --bin mnemo

# --------------- stage 2: runtime ---------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 mnemo \
    && useradd --uid 1000 --gid mnemo --shell /bin/false --create-home mnemo

# Config directory
RUN mkdir -p /etc/mnemo && chown mnemo:mnemo /etc/mnemo

COPY --from=builder /build/target/release/mnemo /usr/local/bin/mnemo
COPY mnemo.yaml /etc/mnemo/mnemo.yaml

RUN chown mnemo:mnemo /etc/mnemo/mnemo.yaml

ENV MNEMO_CONFIG=/etc/mnemo/mnemo.yaml

EXPOSE 8080

USER mnemo

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["mnemo"]
