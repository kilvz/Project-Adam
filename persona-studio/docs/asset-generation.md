# Asset Generation System

## Overview

The Studio asset generation system provides automatic creation of additional assets based on persona `synthesized.md` files. This system detects when personas are updated and can generate various types of assets like prompt-ready versions, platform adaptations, voice configurations, and more.

## How It Works

### 1. Asset Status Tracking

Each persona folder contains a `.assets_status.json` file that tracks:

- **Last synthesized update**: When the `synthesized.md` file was last modified
- **Last assets generation**: When assets were last generated  
- **Pending assets**: List of assets that need to be generated
- **Generated assets**: List of successfully generated assets
- **Generation flags**: Status flags for each asset type
- **Metadata**: Additional information about the persona and generation

### 2. Trigger Detection

The system detects asset generation needs through multiple methods:

#### File Modification Detection
- Compares `synthesized.md` modification time with last asset generation time
- Automatically marks assets as pending when synthesized content is newer

#### README Trigger Markers
Users can add HTML comments to the persona's README.md to trigger asset generation:

```html
<!-- GENERATE:prompt_ready -->
<!-- GENERATE:platform_adaptations -->
<!-- GENERATE:voice_clone -->
<!-- GENERATE:image_avatar -->
<!-- GENERATE:chatbot_config -->
<!-- GENERATE:api_endpoint -->
```

#### Manual Status Updates
Assets can be manually marked as pending through the status API.

### 3. Asset Types

The system supports the following asset types:

- **prompt_ready**: Condensed version optimized for AI prompts
- **platform_adaptations**: Platform-specific versions (Discord, Telegram, etc.)
- **voice_clone**: Voice synthesis configuration
- **image_avatar**: Avatar generation parameters
- **chatbot_config**: Chatbot deployment configuration  
- **api_endpoint**: API integration settings

## Usage

### Command Line Monitoring

Use the asset monitor tool to check and generate assets:

```bash
# Run once to check all personas
go run cmd/asset-monitor/main.go -once

# Check specific persona
go run cmd/asset-monitor/main.go -persona "David Attenborough" -once

# Continuous monitoring (every 30 seconds)
go run cmd/asset-monitor/main.go -interval 30s

# Debug mode
go run cmd/asset-monitor/main.go -log debug -once
```

### Programmatic Usage

```go
import "github.com/twin2ai/studio/internal/assets"

// Create monitor and generator
monitor := assets.NewMonitor(".", logger)
generator := assets.NewGeneratorService(".", logger)

// Register callbacks
generator.RegisterAllCallbacks(monitor)

// Scan for triggers
ctx := context.Background()
triggers, err := monitor.ScanForTriggers(ctx)
if err != nil {
    return err
}

// Process triggers
err = monitor.ProcessTriggers(ctx, triggers)
```

### Manual Asset Status Management

```go
statusManager := assets.NewStatusManager(".")

// Mark asset as pending
err := statusManager.MarkAssetPending("David Attenborough", assets.AssetTypePromptReady)

// Mark asset as generated
err := statusManager.MarkAssetGenerated("David Attenborough", assets.AssetTypePromptReady)

// Check if assets need generation
needs, assetTypes, err := statusManager.NeedsAssetGeneration("David Attenborough")
```

## File Structure

```
personas/
├── david_attenborough/
│   ├── .assets_status.json          # Asset generation status
│   ├── README.md                    # Includes trigger examples
│   ├── synthesized.md               # Source content for assets
│   ├── raw/                         # Raw AI outputs
│   ├── prompt_ready.md              # Generated asset
│   ├── voice_clone.md               # Generated asset
│   ├── image_avatar.md              # Generated asset
│   ├── chatbot_config.json          # Generated asset
│   ├── api_endpoint.md              # Generated asset
│   └── platforms/                   # Generated platform adaptations
│       ├── discord.md
│       ├── telegram.md
│       └── slack.md
```

## Asset Status JSON Format

```json
{
  "persona_name": "David Attenborough",
  "last_synthesized_update": "2025-07-13T10:30:00Z",
  "last_assets_generation": "2025-07-13T10:25:00Z",
  "pending_assets": ["prompt_ready", "voice_clone"],
  "generated_assets": ["platform_adaptations"],
  "asset_generation_flags": {
    "prompt_ready": false,
    "platform_adaptations": true,
    "voice_clone": false
  },
  "metadata": {
    "created_by": "studio",
    "created_at": "2025-07-13T10:20:00Z",
    "regenerated": "false"
  }
}
```

## Extending the System

### Adding New Asset Types

1. Add new asset type to `assets/status.go`:
```go
const (
    AssetTypeCustom AssetType = "custom_asset"
)
```

2. Implement generator in `assets/generators.go`:
```go
func (g *GeneratorService) GenerateCustom(ctx context.Context, personaName string, assetType AssetType) error {
    // Implementation here
}
```

3. Register callback:
```go
monitor.RegisterCallback(AssetTypeCustom, generator.GenerateCustom)
```

### Custom Trigger Detection

Extend the monitor to detect custom triggers:

```go
func (m *Monitor) checkCustomTriggers(personaName string) ([]AssetType, error) {
    // Custom trigger detection logic
}
```

## Integration with Studio Pipeline

The asset generation system integrates with the main Studio pipeline:

1. **Persona Creation**: When Studio creates new personas, it initializes the `.assets_status.json` file
2. **Updates**: When personas are updated, the synthesized timestamp is updated
3. **README Generation**: Studio generates README files with trigger examples
4. **Monitoring**: External monitoring can be set up to automatically generate assets

## Best Practices

1. **Asset Status Management**: Always update asset status when generation completes
2. **Error Handling**: Handle generator failures gracefully, don't mark as completed on error
3. **Incremental Generation**: Only regenerate assets when synthesized content changes
4. **Resource Management**: Consider rate limiting and resource usage for asset generation
5. **Monitoring**: Set up monitoring to detect and resolve asset generation failures

## Limitations

- Asset generation is currently placeholder-based and needs real implementation
- No automatic cleanup of old generated assets
- No version history for generated assets
- No distributed generation support

## Future Enhancements

- Real AI-powered asset generation
- Version control for generated assets
- Distributed asset generation
- Asset validation and quality checks
- Automated testing of generated assets
- Integration with external asset generation services