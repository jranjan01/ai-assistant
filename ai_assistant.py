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

MAX_STEPS = 10

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]


def stream_call(messages):
    """Stream the AI's response. Returns (content, tool_calls)."""
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        tools=TOOLS,
        stream=True,
    )

    full_content = ""
    tool_calls = {}  # index -> accumulated tool call
    label_printed = False

    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 1. content streaming
            if delta.content:
                if not label_printed:  # print the label (Assistant) only once, before the first content fragment
                    print("\n🤖 Assistant: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_content += delta.content

            # 2. tool call streaming: merge fragments by index
            for tc in delta.tool_calls or []:
                obj = tool_calls.setdefault(
                    tc.index,
                    {"id": "", "function": {"name": "", "arguments": ""}},
                )
                if tc.id:
                    obj["id"] = tc.id
                if tc.function:
                    if tc.function.name and not obj["function"]["name"]:
                        obj["function"]["name"] = (
                            tc.function.name
                        )  # name arrives whole, set once
                    if tc.function.arguments:
                        obj["function"]["arguments"] += (
                            tc.function.arguments
                        )  # arguments may arrive in fragments, so we will concatenate them

    except KeyboardInterrupt:
        print("\n stopped by you.")
        tool_calls = {}  # discard any tool calls if the user interrupts

    except Exception as e:  # noqa: BLE001
        print(f"\nError during streaming: {e}")
        tool_calls = {}  # discard any tool calls if an error occurs

    finally:
        if label_printed:
            print()  # print a newline after the assistant's response

    tool_list = []

    for key in sorted(tool_calls):
        tool_list.append(tool_calls[key])

    return full_content, tool_list


def build_tool_calls(tool_calls):
    return [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            },
        }
        for tc in tool_calls
    ]


while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break
    try:
        conversation.append({"role": "user", "content": user_input})

        for step in range(MAX_STEPS):
            content, tool_calls = stream_call(conversation)

            if not tool_calls:
                # save the final reply to history
                conversation.append({"role": "assistant", "content": content or ""})
                log(f"  ↳ {content}")
                break

            log(f"step {step + 1}: model requested {len(tool_calls)} tool call(s).")

            conversation.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": build_tool_calls(tool_calls),
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"] or "{}"

                try:
                    tool_args = json.loads(raw_args)
                except json.JSONDecodeError as parse_error:
                    result = {"error": f"Invalid JSON arguments: {parse_error}"}
                else:
                    log(f" 🔧 Calling tool: {tool_name} with args: {tool_args}")
                    fn = FUNCTION_NAME_TO_FUNCTION.get(tool_name)
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
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, default=str),
                    }
                )
            # Loop continues:
            # AI receives tool results and may call another tool.
        else:  # else block will run - if the for loop completes without a break, meaning the AI has exceeded the maximum number of steps.
            print(
                "\n🤖 Assistant: Maximum steps exceeded. The AI may be stuck in a loop."
            )
            conversation.append(
                {"role": "assistant", "content": "Maximum steps exceeded."}
            )
            continue

    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        log(f" 🛑 Error: {e!s}")
        print("Error:", str(e))
