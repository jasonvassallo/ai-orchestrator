
import sys
import asyncio
import logging
from src.orchestrator import AIOrchestrator, ModelRegistry

# Configure logging to print to stderr
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s', stream=sys.stderr)

async def test_mlx_reasoning():
    print("--- Starting MLX Reasoning Debug ---")
    
    orch = AIOrchestrator(verbose=True)
    
    model_key = "mlx-ministral-14b-reasoning"
    print(f"\n--- Testing Provider for {model_key} ---")
    
    if model_key in ModelRegistry.MODELS:
        try:
            # force get provider which triggers initialize
            provider = await orch._get_provider(model_key)
            if provider:
                print(f"Successfully retrieved '{model_key}' provider.")
                # Try a simple completion to ensure it actually runs
                print("Attempting simple completion...")
                response = await provider.complete(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=model_key,
                    max_tokens=10
                )
                print(f"Completion result: success={response.success}, content='{response.content[:50]}'...")
            else:
                print(f"Failed to retrieve '{model_key}' provider. Check logs.")
        except Exception as e:
            print(f"Exception getting reasoning provider: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model {model_key} not found in registry.")

if __name__ == "__main__":
    asyncio.run(test_mlx_reasoning())
