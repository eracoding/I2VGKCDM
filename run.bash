# Run a single experiment with a configuration file
python run_generation.py --config experiment_config.yaml --experiment-name mountain_scene

# Run a parameter sweep on the variation_strength parameter
python run_generation.py --config base_config.yaml --sweep --param-name variation_strength --param-values 0.1,0.2,0.3,0.4,0.5 --experiment-name variation_sweep