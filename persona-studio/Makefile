# Makefile for Studio

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GORUN=$(GOCMD) run
GOTEST=$(GOCMD) test
GOFMT=$(GOCMD) fmt
BINARY_NAME=studio
MAIN_PATH=cmd/studio/main.go cmd/studio/batch.go

# Build the application
build:
	$(GOBUILD) -o $(BINARY_NAME) $(MAIN_PATH)

# Run the application
run:
	$(GORUN) $(MAIN_PATH)

# Run batch processing
batch:
	$(GORUN) $(MAIN_PATH) batch $(ARGS)

# Run tests
test:
	$(GOTEST) ./...

# Format code
fmt:
	$(GOFMT) ./...

# Clean build artifacts
clean:
	rm -f $(BINARY_NAME)
	rm -f studio-linux

# Cross-compile for Linux
build-linux:
	GOOS=linux GOARCH=amd64 $(GOBUILD) -o studio-linux $(MAIN_PATH)

# Install dependencies
deps:
	$(GOCMD) mod download

# Run linter (requires golangci-lint)
lint:
	golangci-lint run

.PHONY: build run batch test fmt clean build-linux deps lint