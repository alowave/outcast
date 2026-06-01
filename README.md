# OUTCAST - Obstacle-aware UAV–Terrestrial Communication Architecture Simulation Twin

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Build Status](https://github.com/alowave/outcast/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

<img src="./docs/assets/demo.gif" align="right" alt="UAV-SIM Simulation Running" width="42%" style="margin-left: 25px; margin-bottom: 20px;">

**UAV Network Simulator** for multi-layer UAV communications, including fronthaul and backhaul modeling.

The core platform framework targets highly dynamic link budgets, spatial geometry configurations, and automated metric extraction for urban simulation environments.

### Running the Simulation

The primary entry point for the simulation is the `Orchestrator`. You can run it using:

```bash
python -m src.outcast.orchestrator
```

## Configuration

The project uses **Hydra** for robust configuration management.

### Default Configuration
Default settings are organized in the `conf/` directory:
- `conf/config.yaml`: Global settings and scenario parameters.
- Sub-directories (e.g., `conf/world/`, `conf/fronthaul/`): Layer-specific configurations.

### Command-Line Overrides
You can override any configuration parameter directly from the CLI. Examples:
```bash
# Change the scenario and number of steps
python -m src.outcast.orchestrator scenario_name=scenario_1 total_steps=500

# Change a nested parameter (e.g., UAV battery mass)
python -m src.outcast.orchestrator simulation.uav_battery.mass_kg=5.0
```

### Local Development Overrides
For quick iteration without changing YAML files or typing long CLI commands, you can use the local override mechanism in `src/outcast/orchestrator.py`:
1. Open `src/outcast/orchestrator.py`.
2. Set `USE_LOCAL_OVERRIDES = True`.
3. Modify the values within the `_apply_local_overrides(cfg: OrchestratorCfg)` function.

These values will take precedence over both the YAML files and any CLI arguments.

## Available Scenarios

The simulator currently supports three built-in scenarios, selectable via the `scenario_name` config key:

1.  **`city_of_the_sun`**
2. **`poznan_obstacle_scenario`**
2.  **`scenario_1`**
3.  **`scenario_2`**

## Outputs and Metrics

All simulation results are saved in the `outputs/` directory. Hydra automatically creates a timestamped folder for each run (e.g., `outputs/2026-04-21/14-30-00/`).

Inside each run folder:
- **`.hydra/`**: Contains the full configuration snapshot and overrides used for that run.
- **`metrics/`**: Contains simulation performance data saved as `.npz` files (e.g., fronthaul throughput, SINR, etc.).

Metadata regarding the run (like the final merged config) is automatically preserved by Hydra in the `.hydra` sub-folder.

## Credits

* Project Lead: [Salim Janji](https://github.com/slangooo)
* Contributors: [Ada Rolek](https://github.com/adarolek) and [Ivan Martysevich](https://github.com/alowave)
