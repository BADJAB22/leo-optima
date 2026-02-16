import asyncio
import argparse
import os
from Truth_Optima import TruthOptima, TruthOptimaConfig
from api_interfaces import LLMSimulator, OpenAICompatibleAPI

async def main():
    parser = argparse.ArgumentParser(description="LEO Optima - TruthOptima CLI")
    parser.add_argument("query", type=str, nargs="?", help="The question to ask the AI")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--sim", action="store_true", help="Use simulators instead of real APIs")
    parser.add_argument("--export", "-e", type=str, help="Export the response to a file")
    
    args = parser.parse_args()

    # Configuration
    config = TruthOptimaConfig()
    
    # Model Setup
    if args.sim:
        models = {
            'gpt4': LLMSimulator('GPT-4-Sim'),
            'claude': LLMSimulator('Claude-Sim'),
        }
    else:
        # Try to use OpenAI if key is present
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            models = {
                'gpt4': OpenAICompatibleAPI(model_name="gpt-4", api_key=api_key),
            }
        else:
            print("⚠️ No API keys found in environment. Falling back to simulators.")
            models = {
                'gpt4': LLMSimulator('GPT-4-Sim'),
                'claude': LLMSimulator('Claude-Sim'),
            }

    system = TruthOptima(config=config, models=models)

    if args.interactive:
        print("--- LEO Optima Interactive Mode (type 'exit' to quit) ---")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = await system.ask(user_input)
            system.print_response(response)
    elif args.query:
        response = await system.ask(args.query)
        system.print_response(response)
        if args.export:
            system.export_response(response, args.export)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
