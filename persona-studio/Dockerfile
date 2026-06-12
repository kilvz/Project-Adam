# Build stage
FROM golang:1.21-alpine AS builder

# Install git and ca-certificates
RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o studio cmd/studio/main.go

# Final stage
FROM alpine:latest

# Install git for repository operations
RUN apk --no-cache add git ca-certificates

WORKDIR /app

# Create non-root user
RUN addgroup -g 1000 studio && \
    adduser -D -u 1000 -G studio studio

# Copy binary from builder
COPY --from=builder /app/studio .
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/prompts ./prompts

# Create necessary directories
RUN mkdir -p data logs && chown -R studio:studio /app

USER studio

CMD ["./studio"]