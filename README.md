# VLSI HealingSPA

## Setting Up the Environment
To set up the environment, use the provided `vlsi_env.yaml` Conda configuration file. Run the following command:
```bash
conda env create -f vlsi_env.yaml
```

## Running the Application

### Single GPU Environment
If you are using a single GPU, simply run:
```bash
python main.py
```

### Multi-GPU Environment
For multi-GPU setups, follow these steps:

1. Configure `accelerate` by running:
    ```bash
    accelerate config
    ```
    Refer to [Accelerate Configuration Guide](https://huggingface.co/docs/accelerate/package_reference/cli) for detailed instructions.

2. Launch the training process:
    ```bash
    accelerate launch main.py
    ```

## Running Inference
After training, you can run inference using the following command:
```bash
python inference.py --checkpoint <CHK_PATH>
```
Replace `<CHK_PATH>` with the path to your trained model checkpoint.
