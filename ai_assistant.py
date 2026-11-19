import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import FUNCTION_NAME_TO_FUNCTION, TOOLS
from utils.log import log
from utils.system_prompt import SYSTEM_PROMPT

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
)

MAX_STEPS = 10  # Maximum number of steps for tool calls

conversation = []
conversation.append(
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
)

while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break
    try:
        conversation.append({"role": "user", "content": user_input})

        # chained tool calls: the AI may call multiple tools in a single response,
        # and then receive the results of those tools and call more tools, until it has a final answer.
        for step in range(MAX_STEPS):
            response = client.chat.completions.create(
                model=os.getenv("MODEL"), messages=conversation, tools=TOOLS
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                assistant_reply = msg.content or "I couldn't generate a response."

                conversation.append({"role": "assistant", "content": assistant_reply})
                log(f"  ↳ {assistant_reply}")
                print("\n🤖 Assistant: ", assistant_reply)
                break

            log(f"step {step + 1}: model requested {len(msg.tool_calls)} tool call(s).")

            # Save assistant message containing tool calls

            conversation.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as parse_error:
                    tool_args = None
                    result = {"error": f"Invalid JSON arguments: {parse_error}"}

                fn = FUNCTION_NAME_TO_FUNCTION.get(tool_name)
                log(f" 🔧 Calling tool: {tool_name} with args: {tool_args}")

                if fn is None:
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        result = fn(
                            **tool_args
                        )  # **tool_args will convert {"arg1": "value1"} to arg1="value1"
                    except Exception as tool_error:  # noqa: BLE001
                        result = {"error": str(tool_error)}

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )
            # Loop continues:
            # AI receives tool results and may call another tool.

        else:  # else block will run - if the for loop completes without a break, meaning the AI has exceeded the maximum number of steps.
            print("Assistant: Maximum steps exceeded. The AI may be stuck in a loop.")
            conversation.append(
                {"role": "assistant", "content": "Maximum steps exceeded."}
            )
            continue

    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        log(f" 🛑 Error: {e!s}")
        print("Error:", str(e))
